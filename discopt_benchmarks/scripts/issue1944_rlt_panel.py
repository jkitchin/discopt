#!/usr/bin/env python3
"""#1944: CLAUDE.md section 5 panel for the kernel RLT bundle.

The bundle is ``DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS=1``, ``DISCOPT_RLT_LINEQ=1``
with ``DISCOPT_RLT_LINEQ_MAX=0`` (existing products only) and
``DISCOPT_RLT_INEQ=1``. ``--on-env K=V`` replaces the bundle with other ON-arm
variables and ``--off-env K=V`` gives them explicit OFF values (unset otherwise).
``--base-env K=V`` sets variables in BOTH arms. ``--with-corpus`` adds every
in-repo corpus instance with a proven optimum. #1944 ran
the panel once bare and once with ``--base-env DISCOPT_NATIVE_WARM_START=2``,
because the rows pay only on a node LP that does not start cold (record:
``docs/dev/performance-plan.md`` section 80).

Population: every MINLPLib instance in ``instancedata.csv`` that is nonconvex,
uses no operator beyond polynomial/quadratic (every ``op*`` column empty) with
quadratic terms, has at most ``--max-vars`` variables and ``--max-cons``
constraints, and has an ``=opt=`` in ``minlplib.solu``. That is 225 instances
at the defaults (60/60). No name is hard-coded.

Protocol and checks are those of ``issue1913_pseudocost_panel.py``. Each solve
runs in its own subprocess, and the arm order alternates per instance. A worker
runs both arms of an instance. The checks are cert-clean (no bound across the
oracle, no wrong certified optimum, every incumbent passes ``verify_point`` on a
fresh parse) and net-positive (certificates gained and lost, wall).
``--report`` exits non-zero when nothing was compared, a check failed, or a
certificate was lost.

    python -u discopt_benchmarks/scripts/issue1944_rlt_panel.py --out <dir> \\
        --time-limit 30 --workers 2 [--base-env DISCOPT_NATIVE_WARM_START=2]
    python discopt_benchmarks/scripts/issue1944_rlt_panel.py --out <dir> \\
        --time-limit 30 --report
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue1913_pseudocost_panel as base  # noqa: E402
import scip_wins_screen as screen  # noqa: E402

BUNDLE = {
    "DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS": "1",
    "DISCOPT_RLT_LINEQ": "1",
    "DISCOPT_RLT_LINEQ_MAX": "0",
    "DISCOPT_RLT_INEQ": "1",
}


def build_panel(out: Path, max_vars: int, max_cons: int, corpus: bool = False) -> list[dict]:
    data = screen._fetch(f"{screen.MINLPLIB}/instancedata.csv", out / "instancedata.csv")
    solu = screen._fetch(f"{screen.MINLPLIB}/minlplib.solu", out / "minlplib.solu")
    opt: dict[str, float] = {}
    for ln in Path(solu).read_text().splitlines():
        p = ln.split()
        if len(p) >= 3 and p[0] == "=opt=":
            opt[p[1]] = float(p[2])
    with Path(data).open() as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    ops = [k for k in rows[0] if k.startswith("op")]
    panel = []
    # ``--with-corpus``: every in-repo corpus instance with an ``=opt=`` too, so a
    # graduation run covers the population CLAUDE.md section 5 names.
    in_repo = (
        {p.stem for p in (base.REPO / "python/tests/data/minlplib_nl").glob("*.nl")}
        if corpus
        else set()
    )
    for r in rows:
        if r["name"] in in_repo and r["name"] in opt:
            panel.append({"name": r["name"], "opt": opt[r["name"]], "sense": r["objsense"]})
            continue
        if r["name"] not in opt or r["convex"] != "False" or not r["nvars"]:
            continue
        if int(r["nvars"]) > max_vars or int(r["ncons"]) > max_cons:
            continue
        if any(r[o] for o in ops):
            continue
        if int(r["nquadcons"] or 0) + int(r["nquadfunc"] or 0) == 0:
            continue
        panel.append({"name": r["name"], "opt": opt[r["name"]], "sense": r["objsense"]})
    return panel


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        # Section 6: prove each arm runs what it claims, or crash (a CRASH row).
        want = _arm_env()
        live = {k: os.environ.get(k) for k in want}
        assert all(live[k] == v for k, v in want.items() if v is not None), live
        assert all(live[k] is None for k, v in want.items() if v is None), live
        # ``base.run`` passes the fired-stat key as argv[4] (#1941 added it).
        base._solve_one(sys.argv[2], float(sys.argv[3]), sys.argv[4])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-vars", type=int, default=60)
    ap.add_argument("--max-cons", type=int, default=60)
    ap.add_argument("--base-env", action="append", default=[], metavar="K=V")
    ap.add_argument(
        "--on-env",
        action="append",
        default=[],
        metavar="K=V",
        help="the ON arm's variables (default: the #1944 RLT bundle)",
    )
    ap.add_argument(
        "--off-env",
        action="append",
        default=[],
        metavar="K=V",
        help="explicit OFF values for ON variables (default: unset in the OFF arm)",
    )
    ap.add_argument("--with-corpus", action="store_true")
    ap.add_argument("--fired-stat", default="tree/spatial_score_picks")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        sys.exit(base.report(a.out / f"pc_tl{int(a.time_limit)}.jsonl"))
    for kv in a.base_env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    on_env = dict(kv.partition("=")[::2] for kv in a.on_env) or dict(BUNDLE)
    off_env = dict(kv.partition("=")[::2] for kv in a.off_env)
    os.environ[_ON_KEY] = json.dumps(on_env)
    os.environ[_OFF_KEY] = json.dumps(off_env)
    # ``base.run`` sets ``FLAG`` to the arm ("0"/"1") in each subprocess env. Point
    # it at a marker, and expand that marker into the bundle in the child.
    base.FLAG = "DISCOPT_ISSUE1944_ARM"
    base.__file__ = __file__  # the subprocess re-enters THIS script's ``--_solve``
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "nl").mkdir(exist_ok=True)
    panel = build_panel(a.out, a.max_vars, a.max_cons, a.with_corpus)
    print(f"population {len(panel)} (filter in the module docstring)", flush=True)
    base.run(a.out, panel, a.time_limit, a.workers, base.FLAG, a.fired_stat)


_ON_KEY, _OFF_KEY = "DISCOPT_ISSUE1944_ON_ENV", "DISCOPT_ISSUE1944_OFF_ENV"


def _arm_env() -> dict:
    """The variables this subprocess's arm must carry (``None``: must be unset)."""
    on = json.loads(os.environ.get(_ON_KEY, "null") or "null") or dict(BUNDLE)
    off = json.loads(os.environ.get(_OFF_KEY, "null") or "null") or {}
    if os.environ.get("DISCOPT_ISSUE1944_ARM") == "1":
        return on
    return {k: off.get(k) for k in on}


if __name__ == "__main__":
    for _k, _v in _arm_env().items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
    main()
