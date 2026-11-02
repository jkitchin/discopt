#!/usr/bin/env python3
"""#1917: apply the pre-registered §5 rule for ``DISCOPT_SUBNLP_IMPLIED_BOX``.

The rule is ``docs/dev/data/1917-flag-panel/DISCOPT_SUBNLP_IMPLIED_BOX.prereg.md``.
It was written and committed before any panel data existed, and this script was
committed with it. It reads the GDP panel (``issue1917_gdp_flag_panel.py`` rows)
and the per-rep corpus panels (``flag_onoff_corpus_panel.py`` rows), then prints
V / C / N / K and the outcome.

Exit codes: 0 graduate, 3 retire, 1 the flag stays OFF on a C or K violation,
2 the run is void (V).

    python discopt_benchmarks/scripts/issue1917_implied_box_verdict.py \\
        --gdp <A>.jsonl --corpus <B>.rep0.jsonl <B>.rep1.jsonl [--tree <worktree>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "discopt_benchmarks"))

FLAG = "DISCOPT_SUBNLP_IMPLIED_BOX"
N_GDP_CELLS, N_CORPUS = 48, 74
REPS = 2
REL = 1e-4


def _rows(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _has_inc(rec: dict, obj_key: str) -> bool:
    return "error" not in rec and isinstance(rec.get(obj_key), (int, float))


def _better(a: float, b: float, minimize: bool) -> bool:
    tol = REL * max(1.0, abs(b))
    return a < b - tol if minimize else a > b + tol


def _cells(args):
    """Return ``cells`` mapping key -> {arm: [rec per rep]}, and ``meta``.

    Each record is normalised to inc/obj/cert/nodes/err/minimize/wall/path.
    """
    cells: dict = {}
    meta = {"gdp_rows": 0, "corpus_rows": 0, "feas_checks": 0, "bad_path": [], "c": []}
    import issue1917_gdp_flag_panel as gp

    oracle = gp.gb_oracle()
    for r in _rows(args.gdp):
        if r.get("phase") != "solve" or r.get("flag") != FLAG:
            continue
        meta["gdp_rows"] += 1
        key = ("gdp", r["instance"], r["method"])
        rec = {
            "inc": _has_inc(r, "objective"),
            "obj": r.get("objective"),
            "cert": r.get("verdict") == "solved",
            "nodes": r.get("node_count"),
            "err": "error" in r,
            "minimize": r.get("minimize", True),
            "wall": r.get("wall"),
            "path": r.get("discopt"),
            "rep": r["rep"],
        }
        if rec["inc"] and r.get("in_pyomo") is not None:
            meta["feas_checks"] += 1
        if r.get("verdict") == "incorrect":
            meta["c"].append(f"C1 {key} {r['arm']} rep{r['rep']}: incorrect")
        if gp._bound_crosses(r, oracle.get(r["instance"])):
            meta["c"].append(f"C2 {key} {r['arm']} rep{r['rep']}: bound {r.get('bound')}")
        cells.setdefault(key, {"off": {}, "on": {}})[r["arm"]][r["rep"]] = rec
    for rep, path in enumerate(args.corpus):
        for row in _rows(path):
            meta["corpus_rows"] += 2
            key = ("corpus", row["name"], "")
            for msg in row.get("problems") or []:
                meta["c"].append(f"C2 {key} rep{rep}: {msg}")
            for arm in ("off", "on"):
                a = row[arm]
                rec = {
                    "inc": _has_inc(a, "obj"),
                    "obj": a.get("obj"),
                    "cert": bool(a.get("cert")),
                    "nodes": a.get("nodes"),
                    "err": "error" in a,
                    "minimize": a.get("sense", 1) == 1,
                    "wall": a.get("wall"),
                    "path": a.get("discopt_file"),
                    "rep": rep,
                }
                if a.get("feasible") is not None and arm == "on":
                    meta["feas_checks"] += 1
                cells.setdefault(key, {"off": {}, "on": {}})[arm][rep] = rec
    for key, arms in cells.items():
        for arm, reps in arms.items():
            for rep, rec in reps.items():
                if not rec["path"] or not str(rec["path"]).startswith(str(args.tree)):
                    meta["bad_path"].append((key, arm, rep, rec["path"]))
    return cells, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdp", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path, nargs=REPS)
    ap.add_argument("--tree", type=Path, default=REPO, help="worktree the rows must come from")
    args = ap.parse_args(argv)
    cells, meta = _cells(args)

    # V: validity
    v = []
    if meta["gdp_rows"] != N_GDP_CELLS * 2 * REPS:
        v.append(f"V2 gdp rows {meta['gdp_rows']} != {N_GDP_CELLS * 2 * REPS}")
    if meta["corpus_rows"] != N_CORPUS * 2 * REPS:
        v.append(f"V2 corpus solves {meta['corpus_rows']} != {N_CORPUS * 2 * REPS}")
    if meta["bad_path"]:
        v.append(f"V1 {len(meta['bad_path'])} rows from another tree: {meta['bad_path'][:3]}")
    if meta["feas_checks"] == 0:
        v.append("V3 no ON incumbent was feasibility-checked")
    incomplete = [k for k, a in cells.items() if any(len(a[x]) != REPS for x in a)]
    if incomplete:
        v.append(f"V2 {len(incomplete)} cells missing a rep: {incomplete[:3]}")
    print(
        f"== {FLAG}: {len(cells)} cells, {meta['gdp_rows']} GDP + "
        f"{meta['corpus_rows']} corpus solves"
    )
    print(f"feasibility checks on ON incumbents: {meta['feas_checks']}")
    if v:
        print("VOID:\n  " + "\n  ".join(v))
        return 2

    # C: cert-clean
    c = list(meta["c"])
    for key, a in cells.items():
        for rep in range(REPS):
            if a["off"][rep]["cert"] and not a["on"][rep]["cert"]:
                c.append(f"C3 {key} rep{rep}: certified OFF, not ON")
        if all(a["on"][r]["err"] for r in range(REPS)) and not any(
            a["off"][r]["err"] for r in range(REPS)
        ):
            c.append(f"C4 {key}: ON errors in both reps, OFF in neither")

    # N: net-positive (only effects consistent in both reps count)
    def all_(arm, a, f):
        return all(f(a[arm][r]) for r in range(REPS))

    gained = [
        k
        for k, a in cells.items()
        if all_("on", a, lambda x: x["inc"]) and all_("off", a, lambda x: not x["inc"])
    ]
    lost = [
        k
        for k, a in cells.items()
        if all_("off", a, lambda x: x["inc"]) and all_("on", a, lambda x: not x["inc"])
    ]
    better, worse = [], []
    ratios = []
    for k, a in cells.items():
        both_inc = all_("on", a, lambda x: x["inc"]) and all_("off", a, lambda x: x["inc"])
        any_cert = any(a[arm][r]["cert"] for arm in a for r in range(REPS))
        if both_inc and not any_cert:
            mn = a["on"][0]["minimize"]
            if all(_better(a["on"][r]["obj"], a["off"][r]["obj"], mn) for r in range(REPS)):
                better.append(k)
            elif all(_better(a["off"][r]["obj"], a["on"][r]["obj"], mn) for r in range(REPS)):
                worse.append(k)
        if all_("on", a, lambda x: x["cert"]) and all_("off", a, lambda x: x["cert"]):
            med = {
                arm: statistics.median(a[arm][r]["nodes"] or 0 for r in range(REPS)) for arm in a
            }
            ratios.append((med["on"] + 1) / (med["off"] + 1))
    gm = math.exp(statistics.fmean(math.log(x) for x in ratios)) if ratios else None
    n_inst = len({k[1] for k in gained})
    n = []
    if not (len(gained) >= 3 and n_inst >= 2):
        n.append(f"N1 incumbents gained on {len(gained)} cells / {n_inst} instances (< 3 / 2)")
    if lost:
        n.append(f"N2 incumbents lost on {lost}")
    if len(better) < len(worse):
        n.append(f"N3 ON better on {len(better)} < worse on {len(worse)}")
    if gm is None or gm > 1.05:
        n.append(f"N4 node geomean ratio {gm} over {len(ratios)} cells (> 1.05 or none)")

    # K3: wall
    k3 = []
    for rep in range(REPS):
        ws = {arm: [] for arm in ("off", "on")}
        for a in cells.values():
            for arm in ws:
                w = a[arm][rep]["wall"]
                ws[arm].append(min(w if isinstance(w, (int, float)) else 120.0, 120.0))
        sgm = {
            arm: math.exp(statistics.fmean(math.log(w + 10) for w in ws[arm])) - 10 for arm in ws
        }
        k3.append(sgm["on"] / sgm["off"])
        print(f"rep{rep} SGM wall OFF {sgm['off']:.2f} s, ON {sgm['on']:.2f} s, ratio {k3[-1]:.3f}")

    print(f"gained ({len(gained)}): {gained}")
    print(f"lost ({len(lost)}): {lost}")
    print(f"primal better {len(better)} / worse {len(worse)}: {better} / {worse}")
    print(f"node geomean ON/OFF over {len(ratios)} certified cells: {gm}")
    if c:
        print("NOT CERT-CLEAN (K1: stays OFF, root-cause):\n  " + "\n  ".join(c))
        return 1
    if all(r > 1.25 for r in k3):
        print("K3: wall ratio > 1.25 in both reps -> RETIRE")
        return 3
    if n:
        print("cert-clean, NOT net-positive -> RETIRE:\n  " + "\n  ".join(n))
        return 3
    print("cert-clean and net-positive -> GRADUATE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
