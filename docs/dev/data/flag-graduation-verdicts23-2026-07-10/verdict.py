#!/usr/bin/env python
"""Flag-graduation verdicts 2 & 3 analysis (BR-3 / #581 house pattern).

Criteria per flag per verdict:
- incorrect_count = 0: certified-optimal objective matches oracle (=opt=, or the
  [=bestdual=, =best=] bracket) within abs 1e-4 / rel 1e-3; zero slack.
- oracle cross: dual bound must never cross the known optimum (sense-corrected).
- ZERO certificate losses: OFF optimal -> ON non-optimal is a hard fail.
- objective drift (both-certified) <= abs 1e-4 / rel 1e-3.
- perf: node counts on both-certified instances.

Usage: verdict.py <v2|v3>
"""
import json, math, os, sys

SP = os.path.dirname(os.path.abspath(__file__))

# (kind, best, bestdual)  kind: "opt" or "best"
ORACLE = {
    "nvs21": ("opt", -5.6847825, None),
    "st_e36": ("opt", -246.0, None),
    "nvs09": ("opt", -43.13433692, None),
    "alkyl": ("best", -1.764999646, -1.765024983),
    "tls2": ("opt", 5.3, None),
    "nvs01": ("opt", 12.46966882, None),
    "nvs03": ("opt", 16.0, None),
    "nvs04": ("opt", 0.72, None),
    "nvs05": ("opt", 5.470934108, None),
    "nvs06": ("opt", 1.7703125, None),
    "nvs08": ("opt", 23.44972735, None),
    "nvs10": ("opt", -310.8, None),
    "nvs11": ("opt", -431.0, None),
    "nvs12": ("opt", -481.2, None),
    "nvs13": ("opt", -585.2, None),
    "nvs14": ("opt", -40358.15477, None),
    "nvs15": ("opt", 1.0, None),
    "nvs16": ("opt", 0.703125, None),
    "nvs18": ("opt", -778.4, None),
    "nvs20": ("opt", 230.9221652, None),
    "nvs22": ("opt", 6.05822, None),
    "nvs23": ("opt", -1125.2, None),
    "st_miqp1": ("opt", 281.0, None),
    "st_miqp2": ("opt", 2.0, None),
    "st_miqp3": ("opt", -6.0, None),
    "st_miqp4": ("opt", -4574.0, None),
    "st_miqp5": ("opt", -333.8888889, None),
    "st_test2": ("opt", -9.25, None),
    "st_testgr1": ("opt", -12.8116, None),
    "meanvarx": ("opt", 14.36923211, None),
    "alan": ("opt", 2.925, None),
    "ex5_2_2_case1": ("opt", -400.0, None),
    "ex5_2_2_case2": ("opt", -600.0, None),
    "ex5_2_4": ("opt", -450.0, None),
    "ex7_2_2": ("opt", -0.3888114343, None),
    "ex7_2_3": ("best", 7049.24802, 7049.227807),
    "ex4_1_2": ("opt", -663.5000966, None),
    "ex4_1_3": ("opt", -443.6717047, None),
    "ex4_1_8": ("opt", -16.73889318, None),
    "ex4_1_9": ("opt", -5.508013271, None),
    "ex8_1_1": ("opt", -2.021806783, None),
    "ex8_1_6": ("opt", -10.0860015, None),
    "st_bpaf1a": ("opt", -45.37971014, None),
    "st_bpaf1b": ("opt", -42.9625576, None),
    "st_e29": ("opt", -0.9434705, None),
    "st_e31": ("opt", -2.0, None),
    "gbd": ("opt", 2.2, None),
    "gkocis": ("opt", -1.923098738, None),
}

ABS_TOL, REL_TOL, CROSS_SLACK = 1e-4, 1e-3, 1e-4


def load(verdict, arm):
    with open(os.path.join(SP, f"{verdict}_{arm}.json")) as f:
        return json.load(f)


def within(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def is_optimal(r):
    return str(r.get("status", "")).lower().endswith("optimal")


def sense_sign(r):
    return 1.0 if r.get("sense", "min") == "min" else -1.0


def check_arm(verdict, arm, base):
    on = load(verdict, arm)["results"]
    out = {"incorrect": [], "crosses": [], "cert_losses": [], "cert_gains": [],
           "drifts": [], "node_rows": [], "errors": []}
    for inst, r in on.items():
        b = base[inst]
        kind, best, bestdual = ORACLE[inst]
        s = sense_sign(r) if r.get("sense") else sense_sign(b)
        st = str(r.get("status", ""))
        if st in ("ERROR", "NO_RESULT", "OUTER_TIMEOUT"):
            out["errors"].append((inst, st, str(r.get("error", r.get("stderr", "")))[:150]))
        if is_optimal(r) and r.get("objective") is not None:
            obj = r["objective"]
            if kind == "opt":
                if not within(obj, best):
                    out["incorrect"].append((inst, obj, best))
            else:
                lo, hi = sorted([best, bestdual])
                pad = max(ABS_TOL, REL_TOL * max(abs(lo), abs(hi)))
                if not (lo - pad <= obj <= hi + pad):
                    out["incorrect"].append((inst, obj, (lo, hi)))
        bnd = r.get("bound")
        if bnd is not None and best is not None and math.isfinite(bnd):
            ref = best if kind == "opt" else (bestdual if bestdual is not None else best)
            pad = max(CROSS_SLACK, REL_TOL * abs(ref))
            if s * bnd > s * ref + pad:
                out["crosses"].append((inst, bnd, ref))
        if is_optimal(b) and not is_optimal(r):
            out["cert_losses"].append((inst, b.get("status"), st, b.get("bound"), bnd))
        if not is_optimal(b) and is_optimal(r):
            out["cert_gains"].append((inst, b.get("status"), st))
        if is_optimal(b) and is_optimal(r):
            if not within(r.get("objective"), b.get("objective")):
                out["drifts"].append((inst, b.get("objective"), r.get("objective")))
            out["node_rows"].append((inst, b.get("node_count"), r.get("node_count")))
    return out


def main():
    verdict = sys.argv[1]
    base = load(verdict, "off")["results"]
    print(f"=== {verdict} OFF baseline ===")
    for inst, r in base.items():
        print(f"  {inst:18} {str(r.get('status')):>12} obj={r.get('objective')} "
              f"bnd={r.get('bound')} n={r.get('node_count')} w={r.get('wall')}")
    for arm in ["lu_density_route", "obj_branch_priority", "lift_loose_products"]:
        path = os.path.join(SP, f"{verdict}_{arm}.json")
        if not os.path.exists(path):
            print(f"\n=== {arm}: MISSING ===")
            continue
        res = check_arm(verdict, arm, base)
        print(f"\n=== {verdict} {arm} ===")
        print(f"  errors:       {res['errors']}")
        print(f"  incorrect:    {len(res['incorrect'])} {res['incorrect']}")
        print(f"  oracle-cross: {len(res['crosses'])} {res['crosses']}")
        print(f"  cert-losses:  {len(res['cert_losses'])} {res['cert_losses']}")
        print(f"  cert-gains:   {len(res['cert_gains'])} {res['cert_gains']}")
        print(f"  obj drift:    {len(res['drifts'])} {res['drifts']}")
        tot_off = sum(n for _, n, _ in res["node_rows"] if n and n > 0)
        tot_on = sum(n for _, _, n in res["node_rows"] if n and n > 0)
        changed = [(i, a, c) for i, a, c in res["node_rows"] if a != c]
        engaged = len(changed) > 0
        print(f"  nodes (both-certified): OFF {tot_off} -> ON {tot_on}")
        print(f"  node deltas: {changed}")
        green = (not res["incorrect"] and not res["crosses"] and not res["cert_losses"]
                 and not res["drifts"] and not res["errors"] and engaged)
        print(f"  ENGAGED={engaged}  GREEN={green}")


if __name__ == "__main__":
    main()
