#!/usr/bin/env python3
"""The pre-registered graduate/retire verdict for ``DISCOPT_VUB_CMIR_ROOT`` (#1917 F1).

The rule was fixed and committed before the panel ran
(``docs/dev/data/1917-vub-cmir-panel/README.md``). This script applies it to:

* ``--corpus``: rows from ``flag_onoff_corpus_panel.py --reps R --max-load L``
  (``{"name", "rep", "off", "on", "problems"}``).
* ``--gdp``: rows from ``issue1917_gdp_flag_panel.py --flag DISCOPT_VUB_CMIR_ROOT
  --phase solve --all`` (``phase == "solve"``, one row per instance, arm and rep).

It exits 2 if it compared nothing (CLAUDE.md section 6), 0 for GRADUATE and 1 for
NOT GRADUATE.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

NODE_SHIFT = 10.0  # nodes
WALL_SHIFT = 1.0  # seconds
NODE_BAR = 0.95
WALL_BAR = 1.05
_FINISHED = {"optimal", "infeasible"}


def sgm(vals: list[float], shift: float) -> float:
    return math.exp(sum(math.log(v + shift) for v in vals) / len(vals)) - shift


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _pairs(args) -> list[dict]:
    """One record per (panel, instance, rep) with both arms' status, nodes, wall,
    certificate and the ON arm's installed-row count."""
    out: list[dict] = []
    for r in _rows(args.corpus):
        off, on = r["off"], r["on"]
        out.append(
            {
                "panel": "corpus",
                "name": r["name"],
                "rep": r.get("rep", 0),
                "err": "error" in off or "error" in on,
                "st": (off.get("status"), on.get("status")),
                "cert": (bool(off.get("cert")), bool(on.get("cert"))),
                "nodes": (off.get("nodes"), on.get("nodes")),
                "wall": (off.get("wall"), on.get("wall")),
                "rows": (on.get("stats") or {}).get("vub_cmir/rows", 0.0),
                "problems": r.get("problems", []),
            }
        )
    cells: dict[tuple[str, int], dict[str, dict]] = {}
    for r in _rows(args.gdp):
        if r.get("phase") == "solve" and r.get("flag") == "DISCOPT_VUB_CMIR_ROOT":
            cells.setdefault((r["instance"], r["rep"]), {})[r["arm"]] = r
    for (name, rep), arms in sorted(cells.items()):
        if set(arms) != {"off", "on"}:
            continue
        off, on = arms["off"], arms["on"]
        out.append(
            {
                "panel": "gdp",
                "name": name,
                "rep": rep,
                "err": off["verdict"] == "error" or on["verdict"] == "error",
                "st": (off.get("status"), on.get("status")),
                "cert": (off["verdict"] == "solved", on["verdict"] == "solved"),
                "nodes": (off.get("node_count"), on.get("node_count")),
                "wall": (off.get("wall"), on.get("wall")),
                "rows": (on.get("stats") or {}).get("vub_cmir/rows", 0.0),
                "problems": [
                    f"{a}: incorrect" for a, x in arms.items() if x["verdict"] == "incorrect"
                ]
                + [f"{a}: bound past oracle" for a, x in arms.items() if _gdp_crosses(x)],
            }
        )
    return out


def _gdp_crosses(rec: dict) -> bool:
    from issue1917_gdp_flag_panel import _bound_crosses, gb_oracle

    return _bound_crosses(rec, gb_oracle().get(rec["instance"]))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--gdp", required=True, type=Path)
    args = ap.parse_args(argv)
    pairs = _pairs(args)
    if not pairs:
        print("compared nothing")
        return 2
    reps = sorted({p["rep"] for p in pairs})
    fired = {(p["panel"], p["name"]) for p in pairs if p["rows"] > 0}
    print(f"pairs {len(pairs)} (instance x rep), reps {reps}, fired on {len(fired)} instances:")
    print("   " + " ".join(sorted(f"{a}:{b}" for a, b in fired)))

    # -- bar 1/2: cert-clean, no certification regression ------------------------
    problems = [(p["panel"], p["name"], p["rep"], p["problems"]) for p in pairs if p["problems"]]
    lost = [(p["panel"], p["name"], p["rep"]) for p in pairs if p["cert"][0] and not p["cert"][1]]
    gained = [(p["panel"], p["name"], p["rep"]) for p in pairs if p["cert"][1] and not p["cert"][0]]
    errors = [(p["panel"], p["name"], p["rep"]) for p in pairs if p["err"]]
    print(f"\ncert-clean: {len(problems)} pairs with a problem {problems}")
    print(f"certification lost ON (per rep): {lost}")
    print(f"certification gained ON (per rep): {gained}")
    print(f"pairs with an error in either arm (excluded from ratios): {errors}")

    # -- bar 3: node SGM on fired instances that finished in both arms ------------
    node_ratio, node_ratio_all, wall_ratio = [], [], []
    node_n = 0
    for rep in reps:
        rp = [p for p in pairs if p["rep"] == rep and not p["err"]]
        fin = [
            p
            for p in rp
            if (p["panel"], p["name"]) in fired
            and all(_num(v) for v in p["nodes"])
            and (p["st"][0] in _FINISHED and p["st"][1] in _FINISHED)
        ]
        allf = [
            p for p in rp if (p["panel"], p["name"]) in fired and all(_num(v) for v in p["nodes"])
        ]
        w = [p for p in rp if all(_num(v) for v in p["wall"])]
        node_n += len(fin)
        if fin:
            node_ratio.append(
                sgm([p["nodes"][1] for p in fin], NODE_SHIFT)
                / sgm([p["nodes"][0] for p in fin], NODE_SHIFT)
            )
        if allf:
            node_ratio_all.append(
                sgm([p["nodes"][1] for p in allf], NODE_SHIFT)
                / sgm([p["nodes"][0] for p in allf], NODE_SHIFT)
            )
        if w:
            wall_ratio.append(
                sgm([p["wall"][1] for p in w], WALL_SHIFT)
                / sgm([p["wall"][0] for p in w], WALL_SHIFT)
            )
        print(
            f"rep {rep}: node SGM ON/OFF (fired, both finished, n={len(fin)}) "
            f"{node_ratio[-1] if fin else float('nan'):.4f} | (fired, all, n={len(allf)}) "
            f"{node_ratio_all[-1] if allf else float('nan'):.4f} | wall SGM ON/OFF "
            f"(all, n={len(w)}) {wall_ratio[-1] if w else float('nan'):.4f}"
        )
        for panel in ("corpus", "gdp"):
            wp = [p for p in w if p["panel"] == panel]
            if wp:
                r = sgm([p["wall"][1] for p in wp], WALL_SHIFT) / sgm(
                    [p["wall"][0] for p in wp], WALL_SHIFT
                )
                print(f"        wall SGM ON/OFF {panel} (n={len(wp)}) {r:.4f}")

    def ms(v):
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else float("nan"))

    if not node_ratio or not wall_ratio:
        print("no node or wall comparison executed")
        return 2
    nm, nsd = ms(node_ratio)
    wm, wsd = ms(wall_ratio)
    am, asd = ms(node_ratio_all) if node_ratio_all else (float("nan"), float("nan"))
    print(f"\nnode SGM ratio (primary) mean {nm:.4f} sd {nsd:.4f} over {len(node_ratio)} reps")
    print(f"node SGM ratio (secondary, incl. time-limited) mean {am:.4f} sd {asd:.4f}")
    print(f"wall SGM ratio mean {wm:.4f} sd {wsd:.4f} over {len(wall_ratio)} reps")
    print(f"executed comparisons: {len(pairs)} pairs, {node_n} node pairs")

    bars = {
        "cert-clean": not problems,
        "no certification regression": not lost,
        f"node SGM <= {NODE_BAR}": nm <= NODE_BAR,
        f"wall SGM <= {WALL_BAR}": wm <= WALL_BAR,
    }
    for k, v in bars.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    verdict = all(bars.values())
    print("VERDICT:", "GRADUATE" if verdict else "NOT GRADUATE (retire or documented opt-out)")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
