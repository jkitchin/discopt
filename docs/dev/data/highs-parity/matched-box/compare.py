"""Matched-box node comparison: HiGHS's node bounds against discopt's on the same boxes.

Inputs:

* the HiGHS dump (``HS_DUMP_NODEBOX=<file>``, patched build, ``highs_nodebox.patch``):
  ``NBE`` records (one per top-level node-LP solve in ``HighsSearch::evaluateNodeImpl``)
  and ``NBS`` records (one per node separation in ``separateAndStoreBasis``), each with
  the node's full local column box;
* discopt's stderr from a run with ``DISCOPT_NODEBOX_FILE=<that file>``: one ``NODEBOX``
  line per record (``crates/discopt-core/src/bnb/milp_driver/highs_env/nodebox.rs``).

Compares, per box (smaller = weaker, minimisation):

* ``NBE``: HiGHS's node-LP objective against ours (a), the node bound of our search's LP
  (root cut rows included) on the same box;
* ``NBS``: HiGHS's LP objective before separation against (a), and after its separation
  against (b), our bound after our node separation and the re-evaluation the dive does.

HiGHS's LP stops at the objective limit (its upper limit from the seeded incumbent):
a solve that hits it (model status 11) is recorded as ``>= limit``; ours is then
compared with the limit (both prune, or ours does not). Run with ``python -I``; exits
non-zero when no box was compared (CLAUDE.md §6).

Usage: python -I compare.py <highs_nodebox.txt> <discopt_stderr> [--jsonl out.jsonl]
"""

from __future__ import annotations

import json
import math
import statistics
import sys

MS_OPTIMAL = 7
MS_INFEASIBLE = 8
MS_OBJ_BOUND = 11


def hx(t: str) -> float:
    return float(t) if t.lstrip("+-") in ("inf", "nan") else float.fromhex(t)


def read_highs(path: str) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or not t[0].startswith("NB"):
                continue
            if t[0] == "NBE":
                r = {
                    "nodes": int(t[2]),
                    "depth": int(t[3]),
                    "status": int(t[4]),
                    "ms": int(t[5]),
                    "obj": hx(t[6]),
                    "rows": int(t[7]),
                    "model_rows": int(t[8]),
                    "limit": hx(t[9]),
                    "lb_in": hx(t[10]),
                }
            elif t[0] == "NBS":
                r = {
                    "nodes": int(t[2]),
                    "status0": int(t[3]),
                    "ms0": int(t[4]),
                    "obj0": hx(t[5]),
                    "rows0": int(t[6]),
                    "status1": int(t[7]),
                    "ms1": int(t[8]),
                    "obj1": hx(t[9]),
                    "rows1": int(t[10]),
                    "model_rows": int(t[11]),
                    "limit": hx(t[12]),
                    "node_lb": hx(t[13]),
                    "dom_infeasible": int(t[14]),
                    "box_tightened": int(t[15]),
                }
            else:
                raise ValueError(f"unknown record {t[0]}")
            key = (t[0], int(t[1]))
            if key in out:
                raise ValueError(f"duplicate HiGHS record {key}")
            out[key] = r
    return out


def read_ours(path: str) -> tuple[dict[tuple[str, int], dict], str, str]:
    out: dict[tuple[str, int], dict] = {}
    begin = end = ""
    with open(path) as f:
        for line in f:
            if line.startswith("NODEBOX_BEGIN"):
                if begin:
                    raise ValueError("two NODEBOX_BEGIN lines: the harness fired twice")
                begin = line.strip()
            elif line.startswith("NODEBOX_END"):
                end = line.strip()
            elif line.startswith("NODEBOX "):
                kv = dict(p.split("=", 1) for p in line.split()[1:])
                key = (kv["tag"], int(kv["id"]))
                if key in out:
                    raise ValueError(f"duplicate discopt record {key}")
                out[key] = kv
    if not begin or not end:
        raise ValueError("no complete NODEBOX_BEGIN/END block in the discopt log")
    return out, begin, end


def num(s: str | None) -> float | None:
    if s is None:
        return None
    if s.startswith("Some("):
        s = s[5:-1]
    if s == "None":
        return None
    return float(s)


def tie(a: float, b: float) -> bool:
    if math.isinf(a) and math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def highs_value(obj: float, ms: int, limit: float) -> tuple[float, str]:
    """HiGHS's LP value and how to read it: exact, at/above the limit, or infeasible."""
    if ms == MS_OPTIMAL:
        return obj, "exact"
    if ms == MS_OBJ_BOUND:
        return limit, "ge_limit"
    if ms == MS_INFEASIBLE:
        return math.inf, "infeasible"
    return obj, f"ms{ms}"


def classify(ours: float, hv: float, how: str) -> str:
    """'weaker' / 'equal' / 'stronger' for ours against HiGHS (minimisation)."""
    if how == "ge_limit":
        return "equal" if ours >= hv or tie(ours, hv) else "weaker"
    if tie(ours, hv):
        return "equal"
    return "stronger" if ours > hv else "weaker"


def summarize(name: str, rows: list[dict], ours_key: str, hs_key: str) -> int:
    rows = [r for r in rows if r.get(ours_key) is not None and r.get(hs_key) is not None]
    n = len(rows)
    if n == 0:
        print(f"{name}: 0 boxes")
        return 0
    cls = [r[f"cls_{ours_key}_{hs_key}"] for r in rows]
    fin = [
        r[ours_key] - r[hs_key]
        for r in rows
        if r[f"how_{hs_key}"] == "exact" and math.isfinite(r[ours_key])
    ]
    weaker = [
        r[ours_key] - r[hs_key]
        for r in rows
        if r[f"cls_{ours_key}_{hs_key}"] == "weaker" and r[f"how_{hs_key}"] == "exact"
    ]
    rel = [
        (r[ours_key] - r[hs_key]) / (r["limit"] - r[hs_key])
        for r in rows
        if r[f"how_{hs_key}"] == "exact" and math.isfinite(r[ours_key]) and r["limit"] > r[hs_key]
    ]
    print(
        f"{name}: {n} boxes; ours weaker {cls.count('weaker')}, equal {cls.count('equal')}, "
        f"stronger {cls.count('stronger')}"
    )
    if fin:
        q = statistics.quantiles(fin, n=4) if len(fin) >= 2 else [fin[0]] * 3
        print(
            f"  ours - HiGHS over {len(fin)} exact HiGHS values: "
            f"median {statistics.median(fin):.2f}, "
            f"q1 {q[0]:.2f}, q3 {q[2]:.2f}, min {min(fin):.2f}, max {max(fin):.2f}"
        )
    if weaker:
        print(f"  among weaker: median shortfall {statistics.median(weaker):.2f}")
    if rel:
        print(
            "  (ours - HiGHS) / (limit - HiGHS), i.e. ours' share of HiGHS's distance to the "
            f"cutoff: median {statistics.median(rel):.3f}"
        )
    return n


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    jsonl = None
    if "--jsonl" in argv:
        jsonl = argv[argv.index("--jsonl") + 1]
        args.remove(jsonl)
    if len(args) != 2:
        print(__doc__)
        return 2
    hs = read_highs(args[0])
    ours, begin, end = read_ours(args[1])
    print(begin)
    print(end)
    missing = sorted(set(hs) - set(ours))
    extra = sorted(set(ours) - set(hs))
    if missing or extra:
        raise SystemExit(f"record sets differ: {len(missing)} HiGHS-only, {len(extra)} ours-only")
    rows: list[dict] = []
    res_counts: dict[str, int] = {}
    if not hs:
        print("FAIL: no HiGHS records, 0 boxes compared", file=sys.stderr)
        return 1
    for key in sorted(hs, key=lambda k: k[1]):
        h, o = hs[key], ours[key]
        a, b = num(o.get("a")), num(o.get("b"))
        r = {
            "tag": key[0],
            "id": key[1],
            "hs_nodes": h["nodes"],
            "limit": h["limit"],
            "res_a": o["res_a"],
            "a": a,
            "rows_a": int(o["rows_a"]) if "rows_a" in o else None,
            "res_b": o.get("res_b"),
            "b": b,
            "rows_b": int(o["rows_b"]) if "rows_b" in o else None,
            "lift": num(o.get("lift")),
            "lo_out": int(o["lo_out"]),
            "hi_out": int(o["hi_out"]),
            "untrusted": int(o.get("untrusted_a", 0)) + int(o.get("untrusted_b", 0)),
        }
        res_counts[f"{key[0]}:{o['res_a']}"] = res_counts.get(f"{key[0]}:{o['res_a']}", 0) + 1
        if key[0] == "NBE":
            r["hs_pre"], r["how_hs_pre"] = highs_value(h["obj"], h["ms"], h["limit"])
            r["hs_rows_pre"] = h["rows"]
        else:
            r["hs_pre"], r["how_hs_pre"] = highs_value(h["obj0"], h["ms0"], h["limit"])
            r["hs_post"], r["how_hs_post"] = highs_value(h["obj1"], h["ms1"], h["limit"])
            r["hs_rows_pre"], r["hs_rows_post"] = h["rows0"], h["rows1"]
            r["hs_box_tightened"] = h["box_tightened"]
            if b is None:
                # The node did not stay open for ours: (b) is (a).
                r["b"] = a
        if a is not None:
            r["cls_a_hs_pre"] = classify(a, r["hs_pre"], r["how_hs_pre"])
        if key[0] == "NBS" and r["b"] is not None:
            r["cls_b_hs_post"] = classify(r["b"], r["hs_post"], r["how_hs_post"])
        rows.append(r)
    print("our evaluation results:", dict(sorted(res_counts.items())))
    inf_ours = [r for r in rows if r["a"] is not None and math.isinf(r["a"]) and r["a"] > 0]
    print(
        f"boxes infeasible for us (propagation or LP, a=inf): {len(inf_ours)}; "
        f"HiGHS boxes with columns outside our global box: "
        f"{sum(1 for r in rows if r['lo_out'] or r['hi_out'])} of {len(rows)}"
    )
    untrusted = sum(1 for r in rows if r["untrusted"])
    print(f"boxes with an untrusted LP on our side: {untrusted}")
    nbe = [r for r in rows if r["tag"] == "NBE"]
    nbs = [r for r in rows if r["tag"] == "NBS"]
    print(
        "HiGHS NBE value kinds:",
        {
            k: sum(1 for r in nbe if r["how_hs_pre"] == k)
            for k in sorted({r["how_hs_pre"] for r in nbe})
        },
    )
    prune_h = sum(1 for r in nbe if r["how_hs_pre"] in ("ge_limit", "infeasible"))
    prune_o = sum(1 for r in nbe if r["a"] is not None and r["a"] >= r["limit"])
    print(f"NBE boxes pruned by the LP (bound >= limit): HiGHS {prune_h}, ours {prune_o}")
    compared = 0
    compared += summarize("NBE  (a) vs HiGHS node LP", nbe, "a", "hs_pre")
    compared += summarize("NBS  (a) vs HiGHS pre-separation LP", nbs, "a", "hs_pre")
    compared += summarize("NBS  (b) vs HiGHS post-separation LP", nbs, "b", "hs_post")
    sep_h = [r["hs_post"] - r["hs_pre"] for r in nbs if r["how_hs_post"] == "exact"]
    sep_o = [r["b"] - r["a"] for r in nbs if r["b"] is not None and math.isfinite(r["b"])]
    if sep_h and sep_o:
        print(
            f"separation gain (post - pre): HiGHS median {statistics.median(sep_h):.2f} "
            f"over {len(sep_h)}; ours median {statistics.median(sep_o):.2f} over {len(sep_o)}"
        )
    rows_h = [r["hs_rows_pre"] for r in rows]
    rows_o = [r["rows_a"] for r in rows if r["rows_a"] is not None]
    print(
        f"LP rows at evaluation: HiGHS median {statistics.median(rows_h)} "
        f"(min {min(rows_h)}, max {max(rows_h)}); ours median {statistics.median(rows_o)} "
        f"(min {min(rows_o)}, max {max(rows_o)})"
    )
    if jsonl:
        with open(jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"compared {compared} box comparisons over {len(rows)} records")
    if compared == 0:
        print("FAIL: 0 boxes compared", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
