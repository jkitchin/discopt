#!/usr/bin/env python3
"""Phase 4 re-profile entry experiment — rank the four structure-loss levers.

Measurement-only (no solver math changed). For each panel instance we measure:
  - DAG node count (arena_len) — the thing CSE/V-segments would shrink
  - CSE duplicate-node potential (lever 1): fraction of arena nodes that are
    content-identical to an earlier node -> hash-consing would dedup them
  - defined-variable count discarded by nl_parser (lever 2): read from the .nl
    "common exprs" header line (b,c,o,c1,o1); >0 means the parser drops sharing
  - unrecognized-quadratic flag (lever 3): objective/constraint has degree-2
    structure that is_quadratic sees but no Q-matrix is extracted
  - nontrivial-orbit flag (lever 4): detect_symmetries finds orbits of size >= 2
  - JAX relaxation compile count + seconds (perf-plan CC5) via solver_stats /
    xla_compile_count, per-node eval time, FBBT sweep time, wall clock

Guardrails: per-instance hard cap ~90 s, panel <= 8, results persisted to
discopt_benchmarks/results/. Run synchronously. JAX_PLATFORMS/x64 set by caller.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import discopt.modeling as dm

SNAP = Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"
REPO_CORPUS = Path(__file__).resolve().parents[2] / "python/tests/data/minlplib_nl"
RESULTS = Path(__file__).resolve().parents[1] / "results"
TIME_CAP = 90.0


def find_nl(name: str) -> Path | None:
    for base in (REPO_CORPUS, SNAP):
        p = base / f"{name}.nl"
        if p.exists():
            return p
    return None


def attr(obj, name):
    a = getattr(obj, name)
    return a() if callable(a) else a


def defined_var_count(nl_path: Path) -> int:
    """Sum of the 'common exprs: b,c,o,c1,o1' header field (defined vars).

    Returns -1 for a binary ('b') .nl (discopt cannot parse it at all)."""
    try:
        with open(nl_path, "rb") as fh:
            if fh.read(1) != b"g":
                return -1
        with open(nl_path, errors="replace") as fh:
            for _ in range(14):
                line = fh.readline()
                if not line:
                    break
                if "common exprs" in line:
                    nums = line.split("#")[0].split()
                    return sum(int(x) for x in nums)
    except Exception:
        return -2
    return 0


def content_hash_walk(repr_):
    """Bottom-up content hashing over the arena to estimate CSE dedup potential.

    The arena is topologically ordered (children have lower ids). Each node gets a
    canonical content key built from its op and its children's *content* hashes
    (not their ids); a node whose key was already seen is a duplicate that a
    content-addressed intern (hash-consing) would collapse.
    Returns (n_nodes, n_duplicate, {type: dup_count})."""
    alen = attr(repr_, "arena_len")
    content = [0] * alen
    seen: dict = {}
    dup = 0
    dup_by_type: Counter = Counter()
    for i in range(alen):
        n = repr_.get_node(i)
        t = n.get("type")
        if t == "variable":
            key = ("var", n["index"])
        elif t == "constant":
            key = ("const", repr(n["value"]))
        elif t == "unary_op":
            key = ("un", n["op"], content[n["arg"]])
        elif t == "binary_op":
            lo, ro = content[n["left"]], content[n["right"]]
            if n["op"] in ("*", "+"):  # commutative -> order-independent key
                lo, ro = sorted((lo, ro))
            key = ("bin", n["op"], lo, ro)
        elif t == "sum_over":
            key = ("sum", tuple(sorted(content[c] for c in n["terms"])))
        else:
            key = ("other", i)  # never dedup unknown node types
        h = hash(key)
        content[i] = h
        if h in seen:
            dup += 1
            dup_by_type[t] += 1
        else:
            seen[h] = i
    return alen, dup, dict(dup_by_type)


def quadratic_status(repr_):
    """Lever 3: degree-check recognition of quadratic structure (no Q extraction)."""
    n_con = attr(repr_, "n_constraints")
    con_q = 0
    for ci in range(n_con):
        try:
            if repr_.is_constraint_quadratic(ci):
                con_q += 1
        except Exception:
            pass
    cls = {}
    try:
        c = repr_.classify_nonlinear_terms()
        for k in ("bilinear", "trilinear", "multilinear", "monomial"):
            cls[k] = len(c.get(k, []))
        cls["general_nl"] = len(c.get("general_nl", c.get("general", [])))
    except Exception:
        pass
    return {
        "obj_quadratic": bool(repr_.is_objective_quadratic()),
        "obj_bilinear": bool(repr_.is_objective_bilinear()),
        "constraints_quadratic": con_q,
        "term_classes": cls,
    }


def measure_static(name: str, nl_path: Path):
    out = {"instance": name, "nl_path": str(nl_path)}
    dv = defined_var_count(nl_path)
    out["defined_vars_discarded"] = dv
    if dv == -1:
        out["load_status"] = "binary_nl_unreadable"
        return out, None
    t0 = time.perf_counter()
    try:
        model = dm.from_nl(str(nl_path))
    except Exception as e:
        out["load_status"] = f"load_error:{type(e).__name__}:{str(e)[:80]}"
        return out, None
    out["load_seconds"] = round(time.perf_counter() - t0, 4)
    r = model._nl_repr
    alen, dup, dup_by_type = content_hash_walk(r)
    out["load_status"] = "ok"
    out["dag_nodes"] = alen
    out["n_vars"] = attr(r, "n_vars")
    out["n_constraints"] = attr(r, "n_constraints")
    out["cse_duplicate_nodes"] = dup
    out["cse_duplicate_frac"] = round(dup / alen, 4) if alen else 0.0
    out["cse_duplicate_by_type"] = dup_by_type
    out["quadratic"] = quadratic_status(r)
    t0 = time.perf_counter()
    try:
        sym = r.detect_symmetries()
        out["symmetry"] = {
            "orbits_found": sym.get("orbits_found", 0),
            "total_orbit_members": sym.get("total_orbit_members", 0),
            "variables_examined": sym.get("variables_examined", 0),
            "detect_seconds": round(time.perf_counter() - t0, 5),
        }
    except Exception as e:
        out["symmetry"] = {"error": f"{type(e).__name__}:{str(e)[:60]}"}
    try:
        t0 = time.perf_counter()
        r.fbbt()
        out["fbbt_one_sweep_seconds"] = round(time.perf_counter() - t0, 5)
    except Exception as e:
        out["fbbt_one_sweep_seconds"] = f"err:{type(e).__name__}"
    return out, model


def measure_solve(model, oracle):
    out = {}
    t0 = time.perf_counter()
    try:
        res = model.solve(time_limit=TIME_CAP, gap=1e-4)
    except Exception as e:
        out["solve_status"] = f"solve_error:{type(e).__name__}:{str(e)[:80]}"
        out["wall_seconds"] = round(time.perf_counter() - t0, 3)
        return out
    wall = time.perf_counter() - t0
    out["wall_seconds"] = round(wall, 3)
    out["solve_status"] = str(getattr(res, "status", "?"))
    nodes = getattr(res, "node_count", None) or getattr(res, "nodes", None)
    out["node_count"] = nodes
    out["objective"] = getattr(res, "objective", None)
    out["bound"] = getattr(res, "bound", None) or getattr(res, "best_bound", None)
    xc = getattr(res, "xla_compile_count", None)
    xs = getattr(res, "xla_compile_seconds", None)
    out["xla_compile_count"] = xc
    out["xla_compile_seconds"] = xs
    if xs is not None and wall:
        out["xla_compile_frac_of_wall"] = round(xs / wall, 3)
    if nodes:
        out["seconds_per_node"] = round(wall / nodes, 5)
    stats = getattr(res, "solver_stats", None) or {}
    out["solver_stats"] = {
        k: v
        for k, v in stats.items()
        if any(k.startswith(p) for p in ("reduce/", "separate/", "cuts/", "compile"))
    }
    if oracle is not None and out["bound"] is not None:
        with contextlib.suppress(Exception):
            out["bound_le_oracle"] = bool(out["bound"] <= oracle + 1e-4 * (1 + abs(oracle)))
    return out


ORACLE = {
    "nvs17": -1100.4,
    "ex1252": 128893.8,
    "ex1252a": 128893.8,
    "gear4": 1.6434,
    "st_e38": 7197.727,
}


def main(panel):
    RESULTS.mkdir(exist_ok=True)
    rows = []
    skipped = []
    for name in panel:
        nl = find_nl(name)
        if nl is None:
            skipped.append({"instance": name, "reason": "nl_not_found"})
            print(f"[SKIP] {name}: .nl not found", flush=True)
            continue
        print(f"[RUN ] {name}  ({nl})", flush=True)
        static, model = measure_static(name, nl)
        if static.get("load_status") != "ok":
            static["skipped_solve"] = static.get("load_status")
            rows.append(static)
            skipped.append({"instance": name, "reason": static.get("load_status")})
            print(f"       load_status={static.get('load_status')} (solve skipped)", flush=True)
            continue
        dupf = static["cse_duplicate_frac"] * 100
        print(
            f"       nodes={static['dag_nodes']} dup={static['cse_duplicate_nodes']}"
            f" ({dupf:.1f}%) defvars={static['defined_vars_discarded']}"
            f" obj_quad={static['quadratic']['obj_quadratic']}"
            f" orbits={static['symmetry'].get('orbits_found')}",
            flush=True,
        )
        solve = measure_solve(model, ORACLE.get(name))
        static.update(solve)
        xfrac = solve.get("xla_compile_frac_of_wall")
        print(
            f"       wall={solve.get('wall_seconds')}s nodes={solve.get('node_count')}"
            f" s/node={solve.get('seconds_per_node')} xla={solve.get('xla_compile_count')}"
            f"x/{solve.get('xla_compile_seconds')}s ({xfrac} of wall)",
            flush=True,
        )
        rows.append(static)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    outpath = RESULTS / f"p4_reprofile_structure_loss_{stamp}.json"
    payload = {
        "experiment": "phase4_reprofile_structure_loss",
        "generated_utc": stamp,
        "time_cap_seconds": TIME_CAP,
        "panel": panel,
        "n_run": len([r for r in rows if r.get("load_status") == "ok"]),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "rows": rows,
    }
    outpath.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWROTE {outpath}", flush=True)
    print(f"n_run(ok)={payload['n_run']} n_skipped={payload['n_skipped']}", flush=True)
    return 0


if __name__ == "__main__":
    default_panel = ["nvs17", "ex1252", "ex1252a", "gear4", "st_e38"]
    panel = sys.argv[1:] or default_panel
    sys.exit(main(panel))
