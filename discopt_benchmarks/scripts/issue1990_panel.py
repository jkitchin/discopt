#!/usr/bin/env python3
"""#1990: CLAUDE.md section 5 panels for the two ex1252a flags.

``--flag`` picks the flag under test:

* ``DISCOPT_FBBT_EXTENDED_MUL``: FBBT's backward pass through ``z = a * b``
  divided by a factor only when its interval excluded zero. ON adds the extended
  division, ``a in z / (b minus {0})`` when ``z`` excludes zero
  (``crates/discopt-core/src/presolve/fbbt.rs``, ``extended_mul_preimage``).
* ``DISCOPT_RLT_STRIP_RETRY``: a node whose level-1-RLT LP gives no certified
  verdict is re-solved without the RLT rows before the #671 row filter
  (``python/discopt/_relax/mccormick_lp.py``).

Population: the instances ``scip_wins_screen.py`` selects (``--max-vars`` /
``--max-cons``, a proven status on minlplib.org) plus ``--extra`` names.

Two stages:

* ``--screen`` finds the instances the flag can act on and writes
  ``<out>/<flag>/changed.txt``. For the FBBT flag, one subprocess per arm runs
  FBBT on the root box, on each integer variable's two end faces and on both
  midpoint halves of each continuous variable, and an instance is kept when the
  box digests differ. For the RLT flag, the instance is solved once with the flag
  ON and kept when the process-wide retry count is positive. A solve where the retry never ran
  executes the OFF code path, so the others are unchanged by construction.
* The default stage is the solve A/B of ``issue1913_pseudocost_panel.py`` on
  ``--names-file``. Each solve runs in its own subprocess, the arm order
  alternates per instance, and the report gives the cert-clean and net-positive
  checks. ``picks`` is the process-wide retry count
  (``mccormick_lp.RLT_STRIP_TOTALS``), so "rule fired on" counts the instances
  where the retry ran in any relaxer.

    python -u discopt_benchmarks/scripts/issue1990_panel.py --flag F --screen \\
        --out <dir> --max-vars 60 --max-cons 60 --extra ex1252a
    python -u discopt_benchmarks/scripts/issue1990_panel.py --flag F --out <dir> \\
        --names-file <dir>/F/changed.txt --time-limit 30 --workers 2
    python discopt_benchmarks/scripts/issue1990_panel.py --flag F --out <dir> \\
        --time-limit 30 --report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1913_pseudocost_panel as base  # noqa: E402
import scip_wins_screen as screen  # noqa: E402

FLAGS = ("DISCOPT_FBBT_EXTENDED_MUL", "DISCOPT_RLT_STRIP_RETRY")
#: Cap on the boxes one instance's FBBT screen visits (each is one FBBT call).
MAX_BOXES = 400


def _solve_one(nl: str, tl: float) -> None:
    """Subprocess entry: solve, verify the incumbent on a fresh model, print RESULT."""
    import discopt
    import discopt._relax.mccormick_lp as mlp
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(base.REPO)), discopt.__file__
    t0 = time.perf_counter()
    r = from_nl(nl).solve(time_limit=tl, gap_tolerance=base.TOL)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    ss = r.solver_stats or {}
    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "gap_certified": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "nodes": r.node_count,
                "wall": wall,
                # Process-wide, so a retry in any relaxer of the solve counts.
                "picks": mlp.RLT_STRIP_TOTALS["retries"],
                "rlt_strip_certified": mlp.RLT_STRIP_TOTALS["certified"],
                "row_filter_invocations": ss.get("row_filter/invocations"),
                "route": getattr(r, "algorithm_route", None),
                "verify_failed": bad,
            }
        ),
        flush=True,
    )


def _fbbt_digest(nl: str) -> dict:
    """FBBT box digest over root, integer end faces and continuous halves."""
    import discopt
    import numpy as np
    from discopt._relax.disjunctive_config_bound import _box_fbbt
    from discopt._relax.model_utils import flat_variable_bounds
    from discopt.modeling.core import VarType, from_nl

    assert discopt.__file__.startswith(str(base.REPO)), discopt.__file__
    m = from_nl(nl)
    lb, ub = (np.asarray(b, dtype=np.float64) for b in flat_variable_bounds(m))
    kinds = []
    for v in m._variables:
        kinds += [v.var_type] * v.size
    boxes = [(lb.copy(), ub.copy())]
    for j, kind in enumerate(kinds):
        if kind in (VarType.INTEGER, VarType.BINARY):
            for val in (lb[j], ub[j]):
                if np.isfinite(val):
                    lo, hi = lb.copy(), ub.copy()
                    lo[j] = hi[j] = val
                    boxes.append((lo, hi))
        elif np.isfinite(lb[j]) and np.isfinite(ub[j]) and ub[j] > lb[j]:
            mid = 0.5 * (lb[j] + ub[j])
            lo, hi = lb.copy(), ub.copy()
            hi[j] = mid
            boxes.append((lo, hi))
            lo, hi = lb.copy(), ub.copy()
            lo[j] = mid
            boxes.append((lo, hi))
    h = hashlib.sha256()
    n = 0
    for lo, hi in boxes[:MAX_BOXES]:
        tl, tu, crossed = _box_fbbt(m, lo, hi)
        h.update(np.round(tl, 9).tobytes() + np.round(tu, 9).tobytes() + bytes([crossed]))
        n += 1
    return {"digest": h.hexdigest(), "boxes": n}


def _local_nl(name: str) -> Path | None:
    """The in-repo copy of an instance, if the test corpus carries one."""
    for d in ("minlplib_nl", "minlplib"):
        path = base.REPO / "python" / "tests" / "data" / d / f"{name}.nl"
        if path.exists():
            return path
    return None


def _child(flag: str, arm: str, args: list[str], timeout: float, tag: str) -> dict:
    env = dict(os.environ, **{flag: arm})
    try:
        proc = subprocess.run(
            [sys.executable, __file__, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(tag + " ")]
    if not lines:
        return {"err": proc.stderr[-800:], "rc": proc.returncode}
    return json.loads(lines[-1][len(tag) + 1 :])


def run_screen(flag: str, out: Path, panel: list[dict], workers: int, tl: float) -> Path:
    def one(p):
        nl = screen._fetch(f"{screen.MINLPLIB}/nl/{p['name']}.nl", out / "nl" / f"{p['name']}.nl")
        if flag == "DISCOPT_FBBT_EXTENDED_MUL":
            off = _child(flag, "0", ["--_digest", str(nl)], 600, "DIGEST")
            on = _child(flag, "1", ["--_digest", str(nl)], 600, "DIGEST")
            if "err" in off or "err" in on:
                return p["name"], None, f"boxes={off.get('boxes')}"
            return p["name"], off["digest"] != on["digest"], f"boxes={off['boxes']}"
        on = _child(flag, "1", ["--_solve", str(nl), str(tl)], tl + 120, "RESULT")
        if "err" in on:
            return p["name"], None, str(on.get("err"))[-200:]
        return p["name"], (on.get("picks") or 0) > 0, f"retries={on.get('picks')}"

    changed, executed, failed = [], 0, []
    with ThreadPoolExecutor(workers) as ex:
        for name, hit, info in ex.map(one, panel):
            if hit is None:
                failed.append(name)
                tag = "FAILED"
            else:
                executed += 1
                tag = "changed" if hit else "same"
                if hit:
                    changed.append(name)
            print(f"{name:28s} {tag:8s} {info}", flush=True)
    path = out / flag / "changed.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in changed))
    print(f"SCREENED {executed}, changed {len(changed)}, failed {len(failed)}: {failed}")
    if executed == 0:
        sys.exit("ERROR: nothing screened")
    return path


def _named_panel(out: Path, names: set[str]) -> list[dict]:
    """``names`` from minlplib.org's index, proven or not; ``opt`` is the best known
    primal value (``None`` for a proven-infeasible instance)."""
    import csv

    data = screen._fetch(f"{screen.MINLPLIB}/instancedata.csv", out / "meta" / "instancedata.csv")
    panel = []
    for r in csv.DictReader(data.open(), delimiter=";"):
        if r["name"] in names and "'nl'" in r["formats"]:
            try:
                opt = float(r["primalbound"])
            except ValueError:
                opt = None
            panel.append({"name": r["name"], "sense": r["objsense"], "opt": opt})
    for name in sorted(names - {p["name"] for p in panel}):
        # In the test corpus but not in the index: no reference value to check against.
        if _local_nl(name) is None:
            sys.exit(f"ERROR: {name} is neither in the index with an .nl nor in the corpus")
        panel.append({"name": name, "sense": "min", "opt": None})
    return panel


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        assert any(os.environ.get(f) in ("0", "1") for f in FLAGS), "no arm set"
        _solve_one(sys.argv[2], float(sys.argv[3]))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--_digest":
        assert os.environ.get(FLAGS[0]) in ("0", "1"), "no arm set"
        print("DIGEST " + json.dumps(_fbbt_digest(sys.argv[2])), flush=True)
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flag", choices=FLAGS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-vars", type=int, default=60)
    ap.add_argument("--max-cons", type=int, default=60)
    ap.add_argument("--extra", default="", help="comma-separated names added to the screen")
    ap.add_argument("--extra-only", action="store_true", help="screen only the --extra names")
    ap.add_argument("--names-file", type=Path)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--base-env", action="append", default=[], metavar="K=V", help="both arms")
    ap.add_argument("--tag", default="", help="results go to <out>/<flag><tag>")
    ap.add_argument(
        "--allow-unproven",
        action="store_true",
        help="also take --names-file entries without a proven status; their reference is "
        "the best known primal, so only the bound-crossing check is a soundness check",
    )
    a = ap.parse_args()
    for kv in a.base_env:
        k, _, v = kv.partition("=")
        assert k != a.flag, f"--base-env may not set the flag under test {k}"
        os.environ[k] = v
    flag_out = a.out / (a.flag + a.tag)
    if a.report:
        sys.exit(base.report(flag_out / f"pc_tl{int(a.time_limit)}.jsonl"))
    flag_out.mkdir(parents=True, exist_ok=True)
    base.FLAG = a.flag
    base.__file__ = __file__  # the subprocess re-enters THIS script's ``--_solve``
    base._solve_one = _solve_one
    # ``base.run`` fetches each .nl from minlplib.org; an in-repo copy wins, so
    # corpus-only instances (not in the index) can be panelled too.
    _fetch = screen._fetch

    def _fetch_local_first(url: str, dest: Path) -> Path:
        if url.startswith(f"{screen.MINLPLIB}/nl/") and (loc := _local_nl(dest.stem)):
            return loc
        return _fetch(url, dest)

    screen._fetch = _fetch_local_first
    if a.screen:
        panel = [] if a.extra_only else screen.build_panel(a.out, a.max_vars, a.max_cons, None)
        extra = {n for n in a.extra.split(",") if n} - {p["name"] for p in panel}
        if extra:
            panel += (
                _named_panel(a.out, extra)
                if a.allow_unproven
                else screen.build_panel(a.out, 0, 0, extra)
            )
        print(f"screening {len(panel)} instances for {a.flag}", flush=True)
        run_screen(a.flag, a.out, panel, a.workers, a.time_limit)
        return
    if a.names_file is None:
        sys.exit("ERROR: the panel needs --names-file (the screen writes changed.txt)")
    names = {ln.strip() for ln in a.names_file.read_text().split() if ln.strip()}
    panel = (
        _named_panel(a.out, names)
        if a.allow_unproven
        else screen.build_panel(a.out, a.max_vars, a.max_cons, names)
    )
    (flag_out / "nl").mkdir(exist_ok=True)
    base.run(flag_out, panel, a.time_limit, a.workers)


if __name__ == "__main__":
    main()
