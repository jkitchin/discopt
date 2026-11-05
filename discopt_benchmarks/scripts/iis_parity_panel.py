"""IIS baseline for highs-parity P0.6 (role R5): ``compute_iis(include_integrality=True)``.

``docs/dev/highs-parity-plan.md`` P0.6. The MILP IIS engine (``discopt._iis_milp``) is one
persistent HiGHS model; P3 replaces it with a Rust handle and must return the *same* element
sets. This records what HiGHS returns today.

Subcommands:

``capture DIR``  run every IIS-exercising test file under pytest with this module as a plugin:
                 each linear model handed to ``compute_iis`` is exported with
                 ``Model.to_mps`` into ``DIR/<sha256[:16]>.mps`` (de-duplicated by content),
                 with a sidecar naming the tests that built it. A non-linear model (``to_mps``
                 raises) is counted, not exported.
``run DIR --out F``  for each ``.mps``: load it (HiGHS reads it; ``lp_milp_loader`` writes the
                 ``.nl``; ``from_nl``), run ``compute_iis(include_integrality=True)`` in a fresh
                 subprocess ``--reps`` times, and record each run's element sets (constraint
                 indices, bound elements, integrality elements), ``n_solves``,
                 ``proven_irreducible``, ``complete`` and wall. Prints a determinism counter
                 (identical sets across reps) and exits non-zero when nothing ran.

MIPcc27: the plan's optional smallset lives outside the repo; ``run`` records whether it was
found (``--mipcc DIR``), and it is not on this host.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

IIS_TESTS = (
    "python/tests/test_iis.py",
    "python/tests/test_iis_base_checkpoint.py",
    "python/tests/test_iis_cert_check.py",
    "python/tests/test_iis_deadline.py",
    "python/tests/test_iis_exact_confirm.py",
    "python/tests/test_iis_integrality.py",
    "python/tests/test_iis_ladder.py",
    "python/tests/test_iis_parity.py",
    "python/tests/test_iis_pb_core.py",
    "python/tests/test_iis_portfolio.py",
    "python/tests/test_iis_probe_errors.py",
    "python/tests/test_iis_sat_candidate.py",
    "python/tests/test_issue_1618_remainder.py",
    "python/tests/test_issue_1723_iis_probe_warnings.py",
)


# ----------------------------------------------------------------------------- capture
def pytest_configure(config):
    out = Path(os.environ["IIS_CAPTURE_DIR"])  # KeyError on purpose: no silent no-op
    import discopt.infeasibility as inf

    real = inf.compute_iis
    counts = {"calls": 0, "exported": 0, "nonlinear": 0}

    def compute_iis(model, *a, **k):
        counts["calls"] += 1
        try:
            text = model.to_mps()
        except ValueError:
            counts["nonlinear"] += 1
            text = None
        if text is not None:
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            mps, side = out / f"{sha}.mps", out / f"{sha}.json"
            if not mps.exists():
                mps.write_text(text)
                counts["exported"] += 1
            meta = json.loads(side.read_text()) if side.exists() else {"tests": []}
            meta["tests"] = sorted(set(meta["tests"]) | {os.environ.get("PYTEST_CURRENT_TEST", "")})
            side.write_text(json.dumps(meta, indent=1) + "\n")
        (out / "_counts.json").write_text(json.dumps(counts) + "\n")
        return real(model, *a, **k)

    inf.compute_iis = compute_iis


def cmd_capture(a) -> int:
    out = Path(a.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, IIS_CAPTURE_DIR=str(out), PYTHONPATH=str(HERE.parent))
    tests = [t for t in IIS_TESTS if (REPO / t).exists()]
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
           "-p", "iis_parity_panel", *tests]  # fmt: skip
    p = subprocess.run(cmd, cwd=REPO, env=env)
    counts = json.loads((out / "_counts.json").read_text())
    n = len(glob.glob(str(out / "*.mps")))
    print(f"capture: pytest rc={p.returncode} calls={counts['calls']} "
          f"nonlinear={counts['nonlinear']} models={n}", flush=True)  # fmt: skip
    return 0 if n else 1


# ----------------------------------------------------------------------------- run
def iis_signature(res, model) -> dict:
    """The IIS as plain, order-free data: constraint indices and element keys."""
    cons_idx = {id(c): i for i, c in enumerate(model._constraints)}
    return {
        "constraints": sorted(cons_idx[id(c)] for c in res.constraints),
        "bounds": sorted([v.name, side, list(idx)] for v, side, idx in res.bound_elements),
        "integrality": sorted([v.name, list(idx)] for v, idx in res.integrality_elements),
    }


def _worker(path: str, time_limit: float, engine: str = "highs") -> dict:
    import lp_milp_loader as L  # noqa: N812
    from discopt.infeasibility import compute_iis
    from discopt.modeling.core import from_nl

    inst = L.read_instance(path)
    with tempfile.TemporaryDirectory() as td:
        nl = os.path.join(td, "m.nl")
        L.write_nl(inst, nl)
        model = from_nl(nl)
    t = time.perf_counter()
    # ``engine`` (highs-parity P3.2): the P0.6 baseline was captured on HiGHS, the
    # engine before P3.2, so that stays this worker's default.
    res = compute_iis(model, include_integrality=True, time_limit=time_limit, iis_engine=engine)
    wall = time.perf_counter() - t
    return {
        **iis_signature(res, model),
        "n_solves": int(res.n_solves),
        "proven_irreducible": bool(res.proven_irreducible),
        "complete": bool(res.complete),
        "confirm_fallback": bool(res.confirm_fallback),
        "wall": wall,
        "n": inst.n,
        "m": inst.m,
        "n_int": int(inst.is_int.sum()),
    }


def _run_one(path: str, time_limit: float, engine: str = "highs") -> dict:
    p = subprocess.run([sys.executable, str(HERE), "_worker", path, str(time_limit), engine],
                       capture_output=True, text=True, timeout=10 * time_limit + 120)  # fmt: skip
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        tail = (p.stderr.strip().splitlines() or [""])[-1]
        return {"error": f"rc={p.returncode}: {tail[:300]}"}
    return json.loads(lines[0][len("RESULT ") :])


def sets_of(r: dict) -> tuple:
    return (tuple(r["constraints"]), json.dumps(r["bounds"]), json.dumps(r["integrality"]))


def cmd_run(a) -> int:
    files = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    mipcc = {"path": a.mipcc, "found": bool(a.mipcc) and Path(a.mipcc).is_dir()}
    print(f"run: models={len(files)} reps={a.reps} mipcc={mipcc}", flush=True)
    rows, ran, deterministic, errors = [], 0, 0, 0
    for i, f in enumerate(files, 1):
        sha = Path(f).stem
        meta = json.loads(Path(f).with_suffix(".json").read_text())
        reps = [_run_one(f, a.time_limit) for _ in range(a.reps)]
        ok = [r for r in reps if "error" not in r]
        errors += len(reps) - len(ok)
        same = len(ok) == len(reps) and len({sets_of(r) for r in ok}) == 1
        if ok:
            ran += 1
            deterministic += same
        rows.append({"model": sha, "tests": meta["tests"], "reps": reps, "deterministic": same})
        first = ok[0] if ok else reps[0]
        brief = (f"cons={len(first['constraints'])} bnds={len(first['bounds'])} "
                 f"ints={len(first['integrality'])} solves={first['n_solves']} "
                 f"irreducible={first['proven_irreducible']} complete={first['complete']}"
                 if ok else first["error"])  # fmt: skip
        print(f"[iis {i}/{len(files)}] {sha} same={same} {brief}", flush=True)
        Path(a.out).write_text(json.dumps({"mipcc": mipcc, "rows": rows, "ran": ran,
                                           "deterministic": deterministic, "errors": errors},
                                          indent=1) + "\n")  # fmt: skip
    print(f"run: models={len(files)} ran={ran} deterministic={deterministic}/{ran} "
          f"errors={errors}", flush=True)  # fmt: skip
    return 0 if ran else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("dir")
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--time-limit", type=float, default=30.0)
    r.add_argument(
        "--mipcc", default=os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/mipcc27")
    )
    a = p.parse_args(argv)
    return {"capture": cmd_capture, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        res = _worker(sys.argv[2], float(sys.argv[3]), *(sys.argv[4:5] or ["highs"]))
        print("RESULT " + json.dumps(res), flush=True)
    else:
        sys.exit(main())
