"""Does presolve win rows for HiGHS? Default vs presolve off everywhere (incl. sub-MIPs), each
twice, 20 s, 1 thread, on the rows the strip lost. Prints status/time per run (§6: run count)."""
import os, subprocess, sys, time
sys.path.insert(0, "/home/user/discopt/discopt_benchmarks/scripts")
import p42_rens_panel as R
S = os.path.dirname(os.path.abspath(__file__)); HB = f"{S}/hsab/build/bin/highs"
pop = {m["name"]: m for m in R.population()}
ARMS = {"submip_presolve_off": "threads = 1\ntime_limit = 20\nmip_root_presolve_only = true\n",
        "restart_off": "threads = 1\ntime_limit = 20\nmip_allow_restart = false\n",
        "top_presolve_off": "threads = 1\ntime_limit = 20\npresolve = off\n"}
n = 0
for name in sys.argv[1].split(","):
    for arm, opt in ARMS.items():
        of = f"{S}/hsab/opt_noise_{arm}.txt"; open(of, "w").write(opt)
        for rep in range(2):
            env = {a: b for a, b in os.environ.items() if not a.startswith("HS_")}
            t = time.time()
            p = subprocess.run([HB, "--options_file", of, f"/home/user/discopt/{pop[name]['path']}"], capture_output=True, text=True, env=env, timeout=90)
            st = next((l.split()[-1] for l in p.stdout.splitlines() if l.strip().startswith("Status")), "?")
            n += 1
            print(f"{name:16} {arm:17} rep{rep} {st:10} {time.time()-t:5.1f}s", flush=True)
print("runs:", n)
