"""MIPLIB-38 panel: the R1-MILP parity panel of ``docs/dev/highs-parity-plan.md`` (P0.2).

Rebuilds the 38-instance MIPLIB 2017 "easy" panel of ``milp-competitiveness-plan.md``
§0, whose harness (``scratchpad/miplib/``) was never checked in. The size filter that
picked the 38 was not recorded anywhere; the 38 *names* were, in the A14 artifact
``scratchpad/a14/rens_panel.json``, together with each instance's reference optimum. The
panel is that list (``PANEL``), and every reference value there equals the
``miplib2017-v31.solu`` ``=opt=`` value (checked by ``screen``).

Subcommands:

``fetch``   download each instance (``.mps.gz``) and the ``.solu`` oracle from
            miplib.zib.de into ``ref/miplib2017/`` (git-ignored; files are not
            vendored). The first fetch records each file's sha256 in the panel JSON;
            every later fetch verifies it and refuses a mismatch.
``screen``  validate each instance: read with HiGHS (``lp_milp_loader``), solve the
            *converted arrays* with raw HiGHS (``lp_milp_panel.solve_arrays``) and
            require it to reproduce the published optimum; check the ``.nl``
            round-trip loads with the right shape. A mismatch is excluded and listed,
            never dropped silently. Writes the members in ``lp_milp_panel``'s manifest
            format (panel ``MIPLIB``).
``attribute`` (highs-parity P4.1) on the instances a baseline run left uncertified on
            the Rust route, re-solve each on that route (``DISCOPT_LP_MILP_BACKEND=rust``,
            fresh subprocess) at each ``--budgets`` and classify it against the
            published optimum: ``solved`` (certified), ``no-incumbent``, or by the larger
            of ``primal`` = how far the incumbent is from the optimum and ``dual`` = how
            far the bound is, both as ``|opt - v| / (1 + |opt|)``, sense-aware. Records
            the termination reason and wall too (P0.3's rows stopped early).
``run``     delegate to ``lp_milp_panel.py run`` on that manifest with every arm --
            H0 included -- in a fresh subprocess per solve: arms R
            (``DISCOPT_LP_MILP_BACKEND=rust``), H (default route) and H0 (raw highspy),
            matched ``mip_rel_gap``, interleaved with rotated order, route marker
            asserted per row, executed counter, load gate, discopt-under-test marker.

Discipline (CLAUDE.md §6-§10) as in ``lp_milp_panel``: exceptions propagate, progress
is unbuffered per item, the run exits non-zero when it executed nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812
import lp_milp_panel as P  # noqa: E402, N812

BASE_URL = "https://miplib.zib.de/WebData/instances"
SOLU_URL = "https://miplib.zib.de/downloads/miplib2017-v31.solu"
CACHE = REPO / "ref" / "miplib2017"
PANEL_JSON = REPO / "discopt_benchmarks" / "data" / "miplib" / "panel.json"
PANEL_LABEL = "MIPLIB"

#: The 38 instances, in the order of ``scratchpad/a14/rens_panel.json`` (the last
#: checked-in run on the panel of ``milp-competitiveness-plan.md`` §0).
PANEL = (
    "neos-1425699", "blend2", "gr4x6", "enlight8", "p0201", "mik-250-20-75-5",
    "neos-911970", "mik-250-20-75-3", "fiber", "khb05250", "nexp-50-20-1-1",
    "mik-250-20-75-2", "22433", "nexp-50-20-4-2", "gt2", "mik-250-20-75-1", "23588",
    "flugpl", "dcmulti", "enlight_hard", "sp150x300d", "gen", "neos17", "beavma",
    "neos-3611447-jijia", "neos-3610051-istra", "neos-3118745-obra",
    "neos-3611689-kaihu", "neos-3610040-iskar", "neos-5192052-neckar",
    "neos-2624317-amur", "app2-2", "b-ball", "bppc8-02", "supportcase16",
    "supportcase14", "f2gap40400", "gsvm2rl3",
)  # fmt: skip


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310
        f.write(r.read())
    tmp.replace(dest)


def parse_solu(path: Path) -> dict[str, tuple[str, float | None]]:
    """``name -> (tag, value)`` from a MIPLIB ``.solu`` file (``=opt=``, ``=best=``, ...)."""
    out: dict[str, tuple[str, float | None]] = {}
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) >= 2:
            out[t[1]] = (t[0], float(t[2]) if len(t) > 2 else None)
    return out


def _load_panel(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"source": {}, "members": [], "excluded": []}


def _save_panel(path: Path, man: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=1) + "\n")


# ----------------------------------------------------------------------------- fetch
def check_sha(name: str, recorded: str | None, actual: str) -> None:
    """Refuse a file whose sha256 differs from the one recorded on the first fetch."""
    if recorded is not None and recorded != actual:
        raise RuntimeError(f"{name}: sha256 {actual} does not match the recorded {recorded}")


def cmd_fetch(a) -> int:
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    man = _load_panel(Path(a.panel_json))
    src = man.setdefault("source", {})
    shas = src.setdefault("sha256", {})
    solu = cache / "miplib2017-v31.solu"
    if not solu.exists():
        _download(SOLU_URL, solu)
    check_sha(solu.name, shas.get(solu.name), _sha256(solu))
    shas[solu.name] = _sha256(solu)
    for i, name in enumerate(PANEL, 1):
        f = cache / f"{name}.mps.gz"
        if not f.exists():
            _download(f"{BASE_URL}/{name}.mps.gz", f)
        digest = _sha256(f)
        check_sha(name, shas.get(f.name), digest)
        shas[f.name] = digest
        print(f"[fetch {i}/{len(PANEL)}] {name:24s} {f.stat().st_size:>10d} B {digest[:12]}",
              flush=True)  # fmt: skip
    src.update(
        {
            "instances_url": BASE_URL,
            "solu_url": SOLU_URL,
            "cache": os.path.relpath(cache, REPO),
            "panel_origin": "scratchpad/a14/rens_panel.json (38 names + reference optima)",
        }
    )
    _save_panel(Path(a.panel_json), man)
    print(f"fetch: files={len(PANEL)} solu=1 sha256 recorded/verified={len(shas)}", flush=True)
    return 0


# ----------------------------------------------------------------------------- screen
def oracle_mismatch(h0: dict, ref: float, gap: float) -> str | None:
    """Why raw HiGHS on the converted arrays fails to reproduce the published optimum."""
    if h0["status"] != "Optimal":
        return f"unvalidated: array solve status {h0['status']}"
    if h0["objective"] is None or abs(h0["objective"] - ref) > P._tol(ref, gap):
        return f"array objective {h0['objective']} vs published {ref}"
    return None


def cmd_screen(a) -> int:
    P._assert_discopt_under_test()
    print("header", json.dumps(P._header()), flush=True)
    cache = Path(a.cache)
    solu = parse_solu(cache / "miplib2017-v31.solu")
    names = a.instances.split(",") if a.instances else list(PANEL)
    members, excluded, compared = [], [], 0
    with tempfile.TemporaryDirectory() as td:
        for i, name in enumerate(names, 1):
            tag = f"[screen {i}/{len(names)}] {name:24s}"
            f = cache / f"{name}.mps.gz"
            kind, ref = solu.get(name, (None, None))
            if kind != "=opt=" or ref is None:
                excluded.append({"file": str(f), "reason": f"solu entry {kind} {ref}"})
                print(tag, "EXCLUDE solu", kind, ref, flush=True)
                continue
            inst = L.read_instance(str(f))
            t0 = time.perf_counter()
            h0 = P.solve_arrays(inst, a.time_limit, a.gap)
            compared += 1
            why = oracle_mismatch(h0, ref, a.gap)
            if why:
                excluded.append({"file": str(f), "reason": why})
                print(tag, "EXCLUDE", why, flush=True)
                continue
            n_int, nnz = int(inst.is_int.sum()), int(inst.A.nnz)
            oracle = {"status": "Optimal", "objective": ref, "nodes": h0["nodes"]}
            try:
                row = P._roundtrip_member(inst, f, PANEL_LABEL, td, nnz, n_int, oracle, h0["x"])
            except ValueError as e:
                excluded.append({"file": str(f), "reason": f"nl writer: {e}"})
                print(tag, "EXCLUDE nl writer:", e, flush=True)
                continue
            row["name"] = name  # the loader's basename of "x.mps.gz" is "x.mps"
            row["oracle_source"] = "miplib2017-v31.solu =opt="
            row["validation_h0_objective"] = h0["objective"]
            row["validation_h0_wall"] = time.perf_counter() - t0
            row["validation_time_limit"] = a.time_limit
            members.append(row)
            print(tag, f"OK n={inst.n} m={inst.m} nnz={nnz} int={n_int} opt={ref} "
                  f"h0={h0['objective']} nodes={h0['nodes']}", flush=True)  # fmt: skip
    man = _load_panel(Path(a.panel_json))
    if a.instances:  # partial screen: replace only the screened names
        keep = [x for x in man["members"] if x["name"] not in names]
        keep_ex = [x for x in man["excluded"] if Path(x["file"]).name.split(".")[0] not in names]
        members, excluded = keep + members, keep_ex + excluded
    man["members"], man["excluded"] = members, excluded
    man.setdefault("source", {})["screen"] = {"gap": a.gap, "commit": P._header()["commit"]}
    _save_panel(Path(a.panel_json), man)
    for x in excluded:
        print(f"  excluded {Path(x['file']).name}: {x['reason']}", flush=True)
    print(f"screen: members={len(members)} excluded={len(excluded)} compared={compared} "
          f"oracle_mismatches={sum('array objective' in x['reason'] for x in excluded)} "
          f"unvalidated={sum(x['reason'].startswith('unvalidated') for x in excluded)}",
          flush=True)  # fmt: skip
    return 0 if compared > 0 else 1


# ----------------------------------------------------------------------------- attribute
def classify(res: dict, opt: float, maximize: bool, gap: float) -> dict:
    """The P4.1 class of one Rust-route result against the published optimum ``opt``."""
    scale = 1.0 + abs(opt)
    inc, bnd = res.get("objective"), res.get("bound")
    sgn = -1.0 if maximize else 1.0  # sgn * value is in the minimize sense
    primal = None if inc is None else max(0.0, sgn * (inc - opt)) / scale
    dual = None
    if bnd is not None and np.isfinite(bnd):
        dual = max(0.0, sgn * (opt - bnd)) / scale
    if res.get("gap_certified"):
        cls = "solved"
    elif inc is None:
        cls = "no-incumbent"
    elif dual is None or (primal is not None and dual >= primal):
        cls = "dual"
    else:
        cls = "primal"
    return {"class": cls, "primal_short": primal, "dual_short": dual,
            "primal_within_gap": primal is not None and primal <= gap}  # fmt: skip


def cmd_attribute(a) -> int:
    P._assert_discopt_under_test()
    hdr = P._header()
    base = json.loads(Path(a.baseline).read_text())
    unsolved = [r["name"] for r in base["rows"] if not r["R"]["gap_certified"]]
    man = {m["name"]: m for m in _load_panel(Path(a.panel_json))["members"]}
    budgets = [float(b) for b in a.budgets.split(",")]
    print("header", json.dumps(hdr), "unsolved", len(unsolved), "budgets", budgets, flush=True)
    rows, executed = [], 0
    for i, name in enumerate(unsolved, 1):
        mem = man[name]
        inst = L.read_instance(str(REPO / mem["path"]))
        row: dict = {"name": name, "opt": mem["oracle_objective"], "maximize": inst.maximize,
                     "n": inst.n, "m": inst.m, "n_int": int(inst.is_int.sum())}  # fmt: skip
        for tl in budgets:
            ns = argparse.Namespace(time_limit=tl, gap=a.gap, max_nodes=a.max_nodes)
            load = os.getloadavg()[0]
            res = P._run_sub(str(REPO / mem["path"]), "R", ns)
            if "error" in res:
                raise RuntimeError(f"{name} @ {tl}s: {res}")
            P.check_route(name, "R", res)
            executed += 1
            c = classify(res, mem["oracle_objective"], inst.maximize, a.gap)
            row[str(int(tl))] = {**c, "status": res["status"], "objective": res["objective"],
                                 "bound": res["bound"], "nodes": res["nodes"],
                                 "wall": res["wall"], "termination": res.get("termination"),
                                 "load": load}  # fmt: skip
            print(f"[{i}/{len(unsolved)}] {name:22s} {int(tl):>3d}s {c['class']:12s} "
                  f"primal={c['primal_short']} dual={c['dual_short']} "
                  f"status={res['status']} wall={res['wall']:.1f} "
                  f"term={res.get('termination')}", flush=True)  # fmt: skip
        rows.append(row)
        Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows},
                                          indent=1) + "\n")  # fmt: skip
    print(f"attribute: instances={len(rows)} executed={executed}", flush=True)
    return 0 if executed else 1


# ----------------------------------------------------------------------------- run
def cmd_run(a, rest: list[str]) -> int:
    argv = [
        "run", "--manifest", str(a.panel_json), "--h0-subprocess",
        "--time-limit", str(a.time_limit), "--gap", str(a.gap), "--rounds", str(a.rounds),
        "--arms", a.arms,
    ]  # fmt: skip
    if a.instances:
        argv += ["--instances", a.instances]
    if a.load_gate is not None:
        argv += ["--load-gate", str(a.load_gate)]
    if a.out:
        argv += ["--out", a.out]
    if a.resume:
        argv += ["--resume"]
    return P.main(argv + rest)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--panel-json", default=str(PANEL_JSON))
    p.add_argument("--cache", default=str(CACHE))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    s = sub.add_parser("screen")
    s.add_argument("--time-limit", type=float, default=60.0)
    s.add_argument("--gap", type=float, default=P.GAP)
    s.add_argument("--instances", default="")
    t = sub.add_parser("attribute")
    t.add_argument("--baseline", default="docs/dev/data/highs-parity/baseline-dcdac02/miplib.json")
    t.add_argument("--budgets", default="20,60")
    t.add_argument("--gap", type=float, default=P.GAP)
    t.add_argument("--max-nodes", type=int, default=100_000_000)
    t.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--time-limit", type=float, default=20.0)
    r.add_argument("--gap", type=float, default=P.GAP)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--arms", default="R,H,H0")
    r.add_argument("--instances", default="")
    r.add_argument("--load-gate", type=float, default=None)
    r.add_argument("--out", default="")
    r.add_argument("--resume", action="store_true")
    a, rest = p.parse_known_args(argv)
    if a.cmd == "run":
        return cmd_run(a, rest)
    if rest:
        p.error(f"unrecognized arguments: {rest}")
    return {"fetch": cmd_fetch, "screen": cmd_screen, "attribute": cmd_attribute}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
