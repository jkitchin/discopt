"""#2187 graduation panel: HiGHS MILP route on trained/random 2-32-32-1 ReLU nets.

Arms (each a persistent worker process; every instance runs all arms back to back, in
an order rotated per instance, so load drift hits every arm alike):

* ``new``  -- this tree, defaults (both #2187 flags ON);
* ``off``  -- this tree with ``DISCOPT_MILP_TREE_BELOW_ROOT_REPAIR=0`` and
  ``DISCOPT_MILP_REFUTED_PRESOLVE_FREE=0`` (the LP ``kError`` re-solve has no flag);
* ``base`` -- a checkout of the merge base (``--base-pythonpath``), asserted to lack
  the #2187 code.

Instances: every ``*.npz`` in ``--nets`` (keys ``W0..``, ``b0..``, ``lo``, ``hi``;
``W`` is ``(in, out)``), ``--perms`` hidden-neuron permutations of each (the same
function, a different MILP), and ``--n-random`` random nets on the same box.
Formulations: ``relu_bigm``, ``relu_extended``, ``relu_bigm`` + ``ideal_cuts``.

Checks, all against an exact numpy forward pass ``f`` (no solver tolerance enters):

1. a published input reproduces the published objective: ``|f(x) - obj| <= 1e-6 (1+|f|)``;
2. ``bound <= f(x) + 1e-9 (1+|f|)`` for EVERY published input of the same function
   (all arms, formulations and permutations), and ``bound <= min over an 801x801 grid``;
3. a certified ``optimal`` is within the route's gap of the best ``f`` found for that
   function (rel 1e-4 or abs 1e-6): otherwise it is an INCORRECT certificate.

Prints the executed-check count and exits non-zero when it is zero or any check fails.

Usage::

    python -u docs/dev/data/issue-2187/panel.py --nets python/tests/data/issue_2187 \
        --base-pythonpath <worktree>/python --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FORMS = {
    "bigm": dict(strategy="relu_bigm"),
    "ext": dict(strategy="relu_extended"),
    "cuts": dict(strategy="relu_bigm", ideal_cuts=True),
}
FLAGS_OFF = {"DISCOPT_MILP_TREE_BELOW_ROOT_REPAIR": "0", "DISCOPT_MILP_REFUTED_PRESOLVE_FREE": "0"}
MARKER = "_repair_tree_below_root"


def load(path: str) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("W")])
    return (
        [np.asarray(z[f"W{i}"], dtype=np.float64) for i in range(n)],
        [np.asarray(z[f"b{i}"], dtype=np.float64) for i in range(n)],
        np.asarray(z["lo"], dtype=np.float64),
        np.asarray(z["hi"], dtype=np.float64),
    )


def forward(Ws: list[np.ndarray], bs: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    h = np.atleast_2d(X)
    for i, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h.sum(axis=1)


def grid_min(Ws, bs, lo, hi, n=801) -> float:
    g0, g1 = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    X = np.column_stack([g0.ravel(), g1.ravel()])
    return float(min(forward(Ws, bs, X[i : i + 100000]).min() for i in range(0, len(X), 100000)))


# ── worker ───────────────────────────────────────────────────────────────────


def worker() -> None:
    import inspect

    import discopt
    import discopt.modeling as dm
    import discopt.solvers.lp_milp_highs as H
    from discopt.ml import Activation, DenseLayer, NetworkDefinition, NNFormulation

    has = MARKER in inspect.getsource(H)
    print(json.dumps({"hello": discopt.__file__, "marker": has}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        Ws, bs, lo, hi = load(job["path"])
        layers = [DenseLayer(W, b, Activation.RELU) for W, b in zip(Ws[:-1], bs[:-1])]
        layers.append(DenseLayer(Ws[-1], bs[-1], Activation.LINEAR))
        m = dm.Model("surrogate")
        kw = FORMS[job["form"]]
        nn = NNFormulation(m, NetworkDefinition(layers, input_bounds=(lo, hi)), **kw)
        nn.formulate()
        m.minimize(dm.sum(nn.outputs))
        t = time.perf_counter()
        extra = dict(cut_callback=nn.cut_callback) if kw.get("ideal_cuts") else {}
        r = m.solve(time_limit=300, threads=1, deterministic=True, **extra)
        wall = time.perf_counter() - t
        x = None
        if r.objective is not None:
            x = np.asarray(r.value(nn.inputs), dtype=np.float64).ravel().tolist()
        st = r.solver_stats or {}
        print(
            json.dumps(
                {
                    "status": r.status,
                    "objective": r.objective,
                    "bound": r.bound,
                    "gap": r.gap,
                    "gap_certified": bool(getattr(r, "gap_certified", False)),
                    "nodes": r.node_count,
                    "wall": wall,
                    "x": x,
                    "route": (getattr(r, "algorithm_route", "") or "")[-400:],
                    "stats": {
                        k: v
                        for k, v in st.items()
                        if any(
                            s in k
                            for s in (
                                "refut",
                                "tree_bound_below_root",
                                "incumbent_below_root",
                                "presolve_off_error",
                                "presolve_cross_moot",
                                "fixed_int_check_inconclusive",
                            )
                        )
                    },
                }
            ),
            flush=True,
        )


# ── driver ───────────────────────────────────────────────────────────────────


def make_instances(nets: Path, inst: Path, perms: int, n_random: int) -> list[dict]:
    inst.mkdir(parents=True, exist_ok=True)
    out = []

    def save(name, Ws, bs, lo, hi, family):
        p = inst / f"{name}.npz"
        np.savez(
            p,
            lo=lo,
            hi=hi,
            **{f"W{i}": W for i, W in enumerate(Ws)},
            **{f"b{i}": b for i, b in enumerate(bs)},
        )
        out.append({"name": name, "path": str(p), "family": family})

    for src in sorted(nets.glob("*.npz")):
        Ws, bs, lo, hi = load(str(src))
        save(src.stem, Ws, bs, lo, hi, src.stem)
        rng = np.random.default_rng(2187)
        for k in range(perms):
            W, b = [w.copy() for w in Ws], [v.copy() for v in bs]
            for li in range(len(W) - 1):  # permute the hidden units of layer li
                p = rng.permutation(W[li].shape[1])
                W[li], b[li], W[li + 1] = W[li][:, p], b[li][p], W[li + 1][p, :]
            save(f"{src.stem}_perm{k}", W, b, lo, hi, src.stem)
    lo, hi = np.array([-2.0, -1.0]), np.array([2.0, 3.0])
    for s in range(n_random):
        rng = np.random.default_rng(10_000 + s)
        sizes = [2, 32, 32, 1]
        Ws = [rng.normal(0, np.sqrt(2.0 / a), (a, c)) for a, c in zip(sizes[:-1], sizes[1:])]
        bs = [rng.normal(0, 0.1, c) for c in sizes[1:]]
        save(f"random_s{s}", Ws, bs, lo, hi, f"random_s{s}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--nets", type=Path)
    ap.add_argument("--base-pythonpath")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=12)
    ap.add_argument("--only", default="", help="comma-separated instance names to run")
    ap.add_argument("--results", default="results.jsonl", help="file (in --out) to write")
    ap.add_argument("--judge-only", action="store_true", help="judge every results*.jsonl")
    a = ap.parse_args()
    if a.worker:
        worker()
        return 0
    a.out.mkdir(parents=True, exist_ok=True)
    print("uptime:", subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip())
    insts = make_instances(a.nets, a.out / "instances", a.perms, a.n_random)
    if a.judge_only:
        # Chunked runs (a run killed at a wall limit is resumed by --only): every
        # (instance, form) pair counts only if all three arms completed it.
        rows = [json.loads(ln) for f in sorted(a.out.glob("results*.jsonl")) for ln in open(f)]
        n = {}
        for r in rows:
            n[(r["instance"], r["form"])] = n.get((r["instance"], r["form"]), 0) + 1
        rows = [r for r in rows if n[(r["instance"], r["form"])] == 3]
        print(
            f"{len(rows)} rows in complete triples; dropped {sum(v for v in n.values() if v != 3)}"
        )
        return judge(rows, insts, a.out)
    run = [ins for ins in insts if not a.only or ins["name"] in a.only.split(",")]
    here = str(Path(__file__).resolve())
    repo_py = str(Path(__file__).resolve().parents[4] / "python")
    arms = {
        "new": ({}, repo_py, True),
        "off": (FLAGS_OFF, repo_py, True),
        "base": ({}, a.base_pythonpath, False),
    }
    procs = {}
    for arm, (env_add, pp, want_marker) in arms.items():
        env = {k: v for k, v in os.environ.items() if not k.startswith("DISCOPT_MILP_")}
        env.update(env_add, PYTHONPATH=pp)
        p = subprocess.Popen(
            [sys.executable, "-u", here, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
            cwd=a.out,
        )
        hello = json.loads(p.stdout.readline())
        assert hello["hello"].startswith(pp), (arm, hello)
        assert hello["marker"] is want_marker, (arm, hello)
        print(f"arm {arm}: {hello['hello']} marker={hello['marker']}", flush=True)
        procs[arm] = p
    rows = []
    rows_f = open(a.out / a.results, "w")
    order = list(arms)
    for i, ins in enumerate(run):
        for form in FORMS:
            rot = (
                order[(i + list(FORMS).index(form)) % 3 :]
                + order[: (i + list(FORMS).index(form)) % 3]
            )
            for arm in rot:
                p = procs[arm]
                p.stdin.write(json.dumps({"path": ins["path"], "form": form}) + "\n")
                p.stdin.flush()
                res = json.loads(p.stdout.readline())
                row = {
                    "instance": ins["name"],
                    "family": ins["family"],
                    "form": form,
                    "arm": arm,
                    **res,
                }
                rows.append(row)
                rows_f.write(json.dumps(row) + "\n")
                rows_f.flush()
                print(
                    f"{ins['name']:40s} {form:5s} {arm:5s} {res['status']:9s} "
                    f"obj={res['objective']} "
                    f"bound={res['bound']} nodes={res['nodes']} {res['wall']:.1f}s",
                    flush=True,
                )
    for p in procs.values():
        p.stdin.close()
        p.wait()
    return 0 if a.only else judge(rows, insts, a.out)


def judge(rows: list[dict], insts: list[dict], out: Path) -> int:
    nets = {ins["name"]: load(ins["path"]) for ins in insts}
    fam_of = {ins["name"]: ins["family"] for ins in insts}
    checks = 0
    bad: list[str] = []
    # every published input, evaluated exactly, per function family
    fam_vals: dict[str, list[float]] = {}
    for r in rows:
        if r["x"] is None:
            continue
        Ws, bs, lo, hi = nets[r["instance"]]
        x = np.asarray(r["x"])
        assert np.all(x >= lo - 1e-7) and np.all(x <= hi + 1e-7), r["instance"]
        f = float(forward(Ws, bs, x)[0])
        r["f"] = f
        fam_vals.setdefault(fam_of[r["instance"]], []).append(f)
        checks += 1
        if abs(f - r["objective"]) > 1e-6 * (1 + abs(f)):
            bad.append(
                f"{r['instance']} {r['form']} {r['arm']}: f(x)={f!r} != obj {r['objective']!r}"
            )
    gmin = {}
    for fam in sorted(set(fam_of.values())):
        Ws, bs, lo, hi = (
            nets[fam] if fam in nets else nets[next(n for n in nets if fam_of[n] == fam)]
        )
        gmin[fam] = grid_min(Ws, bs, lo, hi)
        fam_vals.setdefault(fam, []).append(gmin[fam])
    for r in rows:
        fam = fam_of[r["instance"]]
        if r["bound"] is not None:
            for v in fam_vals[fam]:
                checks += 1
                if r["bound"] > v + 1e-9 * (1 + abs(v)):
                    bad.append(
                        f"{r['instance']} {r['form']} {r['arm']}: "
                        f"bound {r['bound']!r} > exact value {v!r}"
                    )
                    break
        if r["status"] == "optimal" and r["objective"] is not None:
            best = min(fam_vals[fam])
            checks += 1
            gap = r["f"] - best
            if gap > max(1e-6, 1e-4 * max(abs(best), 1e-10)):
                bad.append(
                    f"INCORRECT {r['instance']} {r['form']} {r['arm']}: "
                    f"certified {r['f']!r}, best {best!r}"
                )
    # per-arm summary
    lines = []
    for arm in ("new", "off", "base"):
        rs = [r for r in rows if r["arm"] == arm]
        cnt = {}
        for r in rs:
            cnt[r["status"]] = cnt.get(r["status"], 0) + 1
        wall = np.array([r["wall"] for r in rs])
        lines.append(
            f"{arm:5s} n={len(rs)} {dict(sorted(cnt.items()))} wall total={wall.sum():.1f}s "
            f"median={np.median(wall):.2f}s"
        )
    # where the arms differ
    diffs = []
    key = lambda r: (r["instance"], r["form"])  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(key(r), {})[r["arm"]] = r
    for k, d in sorted(by.items()):
        sig = {a: (d[a]["status"], d[a]["objective"], d[a]["bound"], d[a]["nodes"]) for a in d}
        if len({v for v in sig.values()}) > 1:
            diffs.append(
                f"{k[0]} {k[1]}: "
                + "; ".join(
                    f"{a} {d[a]['status']} obj={d[a]['objective']} "
                    f"bound={d[a]['bound']} nodes={d[a]['nodes']}"
                    for a in ("base", "off", "new")
                )
            )
    txt = "\n".join(
        [
            "# #2187 panel summary",
            *lines,
            f"executed checks: {checks}",
            f"violations: {len(bad)}",
            *bad,
            f"instance/form pairs where arms differ: {len(diffs)}",
            *diffs,
        ]
    )
    (out / "summary.log").write_text(txt + "\n")
    print(txt, flush=True)
    return 0 if checks and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
