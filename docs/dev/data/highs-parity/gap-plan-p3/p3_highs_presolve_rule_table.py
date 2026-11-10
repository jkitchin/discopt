"""HiGHS presolve per-rule reductions (presolve_rule_logging, patched to allow MIPs) over the
62-row population; sums rows/cols removed per rule and lists, per rule, the rows it acts on.
Prints rows parsed (§6)."""
import collections, os, re, subprocess, sys, json
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__))
tot = collections.defaultdict(lambda: [0, 0]); where = collections.defaultdict(list); n = 0; per = {}
for m in R.population():
    p = subprocess.run([f"{S}/hsab/build/bin/highs", "--options_file", f"{S}/hsab/opt_rulelog.txt", f"/home/user/discopt/{m['path']}"],
                       capture_output=True, text=True, timeout=120)
    t = p.stdout.split("Presolve rule removed")
    if len(t) < 2: continue
    n += 1; per[m["name"]] = {}
    for ln in t[1].splitlines()[2:]:
        mm = re.match(r"(.+?)\s{2,}(\d+)\s+(\d+)\s+(\d+)\s*$", ln)
        if not mm or ln.startswith("Total") or ln.startswith("Original") or ln.startswith("Presolved"):
            if ln.startswith("Total"): break
            continue
        rule, rr, cc = mm.group(1).strip(), int(mm.group(2)), int(mm.group(3))
        tot[rule][0] += rr; tot[rule][1] += cc; per[m["name"]][rule] = (rr, cc)
        if rr + cc: where[rule].append(f"{m['name']}({rr},{cc})")
json.dump(per, open(f"{S}/hs_rule_table.json", "w"))
for rule, (rr, cc) in sorted(tot.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
    print(f"{rule:28} rows {rr:6} cols {cc:6}  {' '.join(where[rule][:8])}")
print("rows parsed:", n)
