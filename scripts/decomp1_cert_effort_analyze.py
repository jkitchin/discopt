"""DECOMP-1 (task #88) analysis: per-instance attribution from res_*.json.

Reads the res_<name>.json files produced by decomp1_cert_effort_drive.py (from
the directory given as argv[1], default: alongside this script) and prints one
JSON row per instance with the lever attribution (A/B/C/primal/certified).
"""

import glob
import json
import os
import sys

SCRATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

ORACLE = {  # minlplib.solu (user sense); amp oracle from HiGHS on the identical matrix
    "clay0303hfsg": 26669.10957,  # =opt=
    "casctanks": 9.163479388,  # =best=
    "tls2": 5.3,  # =opt=
    "nvs05": 5.470934108,  # =opt=
    "nvs09": -43.13433692,  # =opt=
    "tanksize": 1.268643754,  # =opt=
    "st_e36": -246.0,  # =opt=
    "nvs19": -1098.4,  # =opt=
    "ex6_2_5": -70.75207783,  # =best=
    "ex6_2_9": -0.0340661841,  # =best=
    "amp_multi4n": -26.822044702,  # HiGHS-certified optimum of the lifted MILP (min sense)
}

FAIL_KEYS = ("numerical", "error", "iteration_limit", "time_limit", "unbounded", "optimal_nobound")


def relclose(a, b, tol=1e-4):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def analyze(rec):
    name = rec["name"]
    oracle = ORACLE.get(name)
    obj = rec.get("objective")
    bound = rec.get("bound")
    root_bound = rec.get("root_bound")
    root_gap = rec.get("root_gap")
    wall = rec.get("wall_time") or rec.get("harness_wall")

    lp = rec.get("mccormick_lp_impl_counts") or {}
    lp_total = sum(lp.values())
    lp_fail = sum(v for k, v in lp.items() if k in FAIL_KEYS)
    fail_rate = (lp_fail / lp_total) if lp_total else None

    found_opt = relclose(obj, oracle)
    # dual-bound progress toward oracle, user sense (bound <= oracle for min;
    # for max the reported bound is an upper bound >= oracle)
    closed = None
    if bound is not None and root_bound is not None and oracle is not None:
        denom = oracle - root_bound
        if abs(denom) > 1e-12:
            closed = (bound - root_bound) / denom
    stalled_at_root = (
        bound is not None and root_bound is not None and relclose(bound, root_bound, 1e-6)
    )

    # incumbent timeline: first time the incumbent matched the oracle (internal sense)
    t_first_opt = None
    traj = rec.get("traj") or []
    flip = False
    if traj and obj is not None:
        last_inc = traj[-1][2]
        if last_inc is not None and abs(last_inc + obj) < 1e-6 * max(1.0, abs(obj)):
            flip = True
    tgt = None if oracle is None else (-oracle if flip else oracle)
    if tgt is not None:
        for t, _glb, inc in traj:
            if inc is not None and relclose(inc, tgt):
                t_first_opt = t
                break
    stall_frac = None
    if t_first_opt is not None and wall:
        stall_frac = max(0.0, (wall - t_first_opt) / wall)

    ctrs = rec.get("rust_counters") or {}
    pivots = ctrs.get("Phase1Pivots", 0) + ctrs.get("Phase2Pivots", 0)

    # Certification taint: the reported bound is WEAKER than the tree's final
    # global lower bound snapshot -> the tree bound was decertified (nonrigorous
    # sentinel fathom / untrusted node solve) and discarded for a fallback bound.
    tree_glb_final = None
    if traj:
        g = traj[-1][1]
        if g is not None and g not in (float("inf"), float("-inf")):
            tree_glb_final = -g if flip else g
    tree_bound_dropped = False
    if tree_glb_final is not None and bound is not None:
        if flip:  # maximize: user-sense bound is an upper bound; dropped if bound > glb
            tree_bound_dropped = bound - tree_glb_final > 1e-6 * max(1.0, abs(bound))
        else:
            tree_bound_dropped = tree_glb_final - bound > 1e-6 * max(1.0, abs(bound))

    uncertified_zero_gap = (
        rec.get("status") not in ("optimal",)
        and not rec.get("gap_certified")
        and rec.get("gap") is not None
        and rec["gap"] <= 1e-6
    ) or (rec.get("status") == "iteration_limit" and relclose(obj, bound) and obj is not None)

    # --- attribution ---
    reasons = []
    certified = rec.get("status") == "optimal"
    if not certified:
        if obj is None and rec.get("status") in ("time_limit", "node_limit"):
            reasons.append("primal")  # no incumbent at all: primal-heuristic gap
        # A: root_gap > ~20%, or bound stalls at root value
        if (root_gap is not None and root_gap > 0.2) or stalled_at_root:
            reasons.append("A")
        # B: node-LP failure rate > 1% (on a meaningful LP count), a
        # found-but-uncertified zero-gap exit, or a tainted/dropped tree bound
        if (
            (fail_rate is not None and lp_total >= 20 and fail_rate > 0.01)
            or uncertified_zero_gap
            or tree_bound_dropped
        ):
            reasons.append("B")
        # C: relaxation is fine (small root gap) yet the tree/machinery still
        # cannot finish inside the budget
        if (
            (root_gap is not None and root_gap <= 0.2)
            and not stalled_at_root
            and "B" not in reasons
            and obj is not None
        ):
            reasons.append("C")
        if not reasons:
            reasons.append("?")
    attribution = "certified" if certified else "+".join(reasons)

    return {
        "name": name,
        "oracle": oracle,
        "status": rec.get("status"),
        "objective": obj,
        "found_opt": found_opt,
        "bound": bound,
        "root_bound": root_bound,
        "root_gap": root_gap,
        "root_time": rec.get("root_time"),
        "gap": rec.get("gap"),
        "gap_certified": rec.get("gap_certified"),
        "nodes": rec.get("node_count"),
        "wall": wall,
        "lp_total": lp_total,
        "lp_fail": lp_fail,
        "lp_fail_rate": fail_rate,
        "lp_status_counts": lp,
        "engine_counts": rec.get("engine_counts"),
        "pivots": pivots,
        "dense_retries": ctrs.get("LpDenseRetries", 0),
        "bound_closed_frac": closed,
        "tree_glb_final": tree_glb_final,
        "tree_bound_dropped": tree_bound_dropped,
        "stalled_at_root": stalled_at_root,
        "t_first_opt_incumbent": t_first_opt,
        "stall_frac": stall_frac,
        "uncertified_zero_gap": uncertified_zero_gap,
        "attribution": attribution,
    }


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(SCRATCH, "res_*.json"))):
        rec = json.load(open(path))
        rows.append(analyze(rec))
    order = list(ORACLE)
    rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    for r in rows:
        print(json.dumps(r, indent=None, default=str))
        print()


if __name__ == "__main__":
    main()
