"""Which HiGHS presolve rules win issue-2446 / enlight_hard / neos-911970? Default HiGHS (1 thread,
20 s) with one rule switched off via presolve_rule_off = 1 << rule. Prints status, time and the
presolve reduction line per run (§6: run count)."""
import os, subprocess, sys, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); HB = f"{S}/hsab/build/bin/highs"
pop = {m["name"]: m for m in R.population()}
RULES = {"none": None, "ForcingRow": 6, "ForcingCol": 7, "FreeColSubst": 8, "DoubletonEq": 9, "DependentEq": 10,
         "DependentFreeCols": 11, "Aggregator": 12, "ParallelRowsCols": 13, "Sparsify": 14, "Probing": 15,
         "Enumeration": 16, "DualFixing": 17, "ColStuffing": 18, "InitialSweep": 19}
n = 0
for name in sys.argv[1].split(","):
    for rule, bit in RULES.items():
        of = f"{S}/hsab/opt_rule.txt"
        open(of, "w").write("threads = 1\ntime_limit = 20\n" + (f"presolve_rule_off = {1 << bit}\n" if bit is not None else ""))
        env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}
        t = time.time()
        p = subprocess.run([HB, "--options_file", of, f"/home/user/discopt/{pop[name]['path']}"], capture_output=True, text=True, env=env, timeout=90)
        st = next((l.split()[-1] for l in p.stdout.splitlines() if l.strip().startswith("Status")), "?")
        red = next((l.strip() for l in p.stdout.splitlines() if l.strip().startswith("Presolve reductions")), "-")
        n += 1
        print(f"{name:13} -{rule:17} {st:10} {time.time()-t:5.1f}s  {red}", flush=True)
print("runs:", n)
