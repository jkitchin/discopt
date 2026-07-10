"""DECOMP-1 (task #88) driver: run the certification-effort panel sequentially.

Runs each instance via decomp1_cert_effort_run.py in a fresh subprocess with
DISCOPT_PROFILE=1, parses the Rust profile dumps from stderr, and writes one
res_<name>.json per instance next to this script (or set DECOMP1_OUT).

Usage: python scripts/decomp1_cert_effort_drive.py [time_limit] [name1,name2,...]
"""

import json
import os
import re
import subprocess
import sys

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("DECOMP1_OUT", SCRATCH)
NL_DIR = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
RUNNER = os.path.join(SCRATCH, "decomp1_cert_effort_run.py")

PANEL = [
    "clay0303hfsg",
    "casctanks",
    "tls2",
    "nvs05",
    "nvs09",
    "tanksize",
    "st_e36",
    "nvs19",
    "ex6_2_5",
    "ex6_2_9",
]

PHASE_RE = re.compile(r"^\s{2}(\w+)\s+(\d+) calls\s+([\d.]+) ms$")
CTR_RE = re.compile(r"^\s{2}(\w+)\s+(\d+)$")


def parse_profile(stderr_path):
    """Sum phase counts/ms and counters across all DISCOPT_PROFILE dumps."""
    phases: dict = {}
    ctrs: dict = {}
    with open(stderr_path, errors="replace") as fh:
        for line in fh:
            m = PHASE_RE.match(line.rstrip())
            if m:
                name, cnt, ms = m.group(1), int(m.group(2)), float(m.group(3))
                c, t = phases.get(name, (0, 0.0))
                phases[name] = (c + cnt, t + ms)
                continue
            m = CTR_RE.match(line.rstrip())
            if m:
                name, val = m.group(1), int(m.group(2))
                ctrs[name] = ctrs.get(name, 0) + val
    return (
        {k: {"calls": v[0], "ms": round(v[1], 1)} for k, v in phases.items()},
        {k: v for k, v in ctrs.items() if v},
    )


def run_one(name, extra_args, tl):
    out_json = os.path.join(OUT_DIR, f"res_{name}.json")
    err_path = os.path.join(OUT_DIR, f"err_{name}.txt")
    env = dict(os.environ, DISCOPT_PROFILE="1")
    cmd = [sys.executable, RUNNER, "--name", name, "--time-limit", str(tl)] + extra_args
    print(f"=== {name} ===", flush=True)
    with open(err_path, "w") as errf:
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=errf, timeout=tl * 4 + 120, text=True, env=env
            )
            stdout = proc.stdout
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            rc = "timeout"
    rec = {"name": name, "returncode": rc}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON: "):
            rec.update(json.loads(line[len("RESULT_JSON: ") :]))
            break
    else:
        rec["error"] = "no RESULT_JSON"
        rec["stdout_tail"] = (stdout or "")[-2000:]
    phases, ctrs = parse_profile(err_path)
    rec["rust_phases"] = phases
    rec["rust_counters"] = ctrs
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=1)
    print(
        f"  status={rec.get('status')} obj={rec.get('objective')} bound={rec.get('bound')} "
        f"nodes={rec.get('node_count')} certified={rec.get('gap_certified')}",
        flush=True,
    )
    return rec


def main():
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    for name in PANEL:
        if only and name not in only:
            continue
        run_one(name, ["--nl", os.path.join(NL_DIR, f"{name}.nl")], tl)
    if only is None or "amp_multi4n" in (only or []):
        run_one("amp_multi4n", ["--amp-milp"], tl)


if __name__ == "__main__":
    main()
