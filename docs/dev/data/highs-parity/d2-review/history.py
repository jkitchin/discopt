"""Certification history per instance across every checked-in D2 §5 panel (read-only).
Prints OFF/ON certified reps per panel for the rows of interest, plus OFF-arm totals.
Exits 2 if no panel row was read."""
import gzip, json, os, sys
ROOT = "/home/user/discopt/docs/dev/data/highs-parity/d2-analysis/"
PANELS = [  # chronological, from the plan/analysis record
    "rins_panel.json", "r1_improve_dive_panel.json.gz", "reentry_cuts_panel.json.gz",
    "node_gmi_panel.json.gz", "aligned_slacks_panel.json.gz", "bundle_aligned_submip_panel.json.gz",
    "w1_cut_lp_panel_r3.json.gz", "w1_cut_lp_sb_armed_panel_r3.json.gz", "w2_rand_round_panel_r3.json.gz",
    "w5_expel_panel_r3.json.gz", "eng_incr_pricing_panel_r3.json.gz", "w4_cmir_bundle_panel_r3.json.gz",
    "eng_incr_rowprice_panel_r3.json.gz", "w4_cmir_bundle_incr_panel_r3.json.gz",
    "tol_robust_panel_v1_r3.json.gz", "tol_robust_panel_v2_r3.json.gz",
]
ROWS = ["enlight8", "p0548", "mik-250-20-75-1", "mik-250-20-75-2", "mik-250-20-75-3", "mik-250-20-75-5",
        "b-ball", "fiber", "neos-3610051-istra", "neos-3610040-iskar", "neos-3611689-kaihu", "neos-3611447-jijia",
        "bppc8-02", "gesa2", "beavma", "sp150x300d", "nexp-50-20-1-1", "nexp-50-20-4-2", "neos17", "gsvm2rl3",
        "neos-911970", "2122", "issue-2446", "gt2", "khb05250", "dcmulti", "neos-1425699", "blend2", "bell5"]
n = 0
hist = {}
tot = []
for p in PANELS:
    path = ROOT + p
    if not os.path.exists(path):
        print("missing", p); continue
    d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    flag = d["args"]["flag"]
    commit = d["header"].get("commit")
    c = {}
    for r in d["rows"]:
        n += 1
        k = r["name"] if r["panel"] == "MIPLIB" or r["name"] not in ("gt2", "sp150x300d", "dcmulti", "flugpl") else r["name"] + "(M)"
        e = c.setdefault(k, [0, 0, 0, 0, 0, 0])  # off cert, on cert, reps, off inc, on inc, nodes_on
        e[0] += r["off"]["gap_certified"]; e[1] += r["on"]["gap_certified"]; e[2] += 1
        e[3] += r["off"]["objective"] is not None; e[4] += r["on"]["objective"] is not None
        e[5] += r["on"]["nodes"]
    hist[p] = (flag, commit, c)
    tot.append((p, flag, commit, sum(v[0] for v in c.values()), sum(v[1] for v in c.values()), sum(v[2] for v in c.values())))
print(f"rows read: {n}")
if n == 0: sys.exit(2)
print("\npanel totals (OFF cert / ON cert / solves):")
for p, flag, commit, off, on, s in tot:
    print(f"  {p:40} {flag:38} {str(commit)[:8]:8} {off:4} {on:4} {s:4}")
hdr = "instance".ljust(22) + " ".join(f"{p.split('_panel')[0][:9]:>9}" for p in PANELS if p in hist)
print("\nOFF/ON certified reps per panel (3 reps; '-' = absent)\n" + hdr)
for row in ROWS:
    line = row.ljust(22)
    for p in PANELS:
        if p not in hist: continue
        c = hist[p][2]
        e = c.get(row)
        line += f"{(f'{e[0]}/{e[1]}' if e else '-'):>9} "
    print(line)
print("\nnodes (ON, summed over reps) for mik-1 and enlight8 per panel:")
for p in PANELS:
    if p in hist:
        c = hist[p][2]
        print(f"  {p:40} mik-1 {c.get('mik-250-20-75-1',[0]*6)[5]:>8} enlight8 {c.get('enlight8',[0]*6)[5]:>8} istra {c.get('neos-3610051-istra',[0]*6)[5]:>8}")
