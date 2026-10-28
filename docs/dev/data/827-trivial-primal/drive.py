"""Stage 1 screen over the MINLPLib snapshot, then stage 2 OFF/ON solves.
python -u drive.py <stage> <workers> [tl]"""
import csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")
stage, W = sys.argv[1], int(sys.argv[2])

def run(args, timeout):
    try:
        p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "tp_one.py"), *args],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"err": "timeout"}
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    return {"err": (p.stderr.strip().splitlines() or ["no output"])[-1][:300], "rc": p.returncode}

if stage == "screen":
    sizes = {r["name"]: r for r in csv.DictReader(open(f"{B}/problem_sizes.csv"))}
    names = sorted(n for n, r in sizes.items() if int(r["vars"]) <= 5000
                   and os.path.exists(f"{B}/minlplib/nl/{n}.nl"))
    out = open(os.path.join(HERE, "screen.jsonl"), "w")
    def one(n):
        r = run(["screen", f"{B}/minlplib/nl/{n}.nl"], 120); r.setdefault("name", n); return r
    done = 0
    with ThreadPoolExecutor(W) as ex:
        for r in ex.map(one, names):
            out.write(json.dumps(r) + "\n"); out.flush(); done += 1
            if done % 100 == 0: print(f"screen {done}/{len(names)}", flush=True)
    print("SCREEN DONE", done, flush=True)
else:
    tl = sys.argv[3]
    scr = [json.loads(l) for l in open(os.path.join(HERE, "screen.jsonl"))]
    cand = [r["name"] for r in scr if r.get("hits")]
    out = open(os.path.join(HERE, f"solve_tl{tl}.jsonl"), "a")
    def one(n):
        nl = f"{B}/minlplib/nl/{n}.nl"
        res = [run(["solve", nl, "0", tl], float(tl) * 6 + 60)]
        if res[0].get("obj") is None:  # only where the default finds no incumbent can it gain
            res.append(run(["solve", nl, "1", tl], float(tl) * 6 + 60))
        for r in res: r.setdefault("name", n)
        return res
    done = 0
    with ThreadPoolExecutor(W) as ex:
        for res in ex.map(one, cand):
            for r in res: out.write(json.dumps(r) + "\n")
            out.flush(); done += 1
            print(f"solve {done}/{len(cand)} {res[0]['name']} " + " | ".join(
                f"{r.get('flag')}:{r.get('status', r.get('err'))} {r.get('obj')}" for r in res), flush=True)
    print("SOLVE DONE", done, flush=True)
