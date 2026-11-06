"""Isolate DISCOPT_COMPL_BRANCH from its route change (#2167).

Flag ON also bypasses the native spatial kernel, so ON-vs-OFF in the H2 and
non-ReLU panels compares the dichotomy *and* a different tree. Here both arms
run on the Python tree (``DISCOPT_NATIVE_SPATIAL_KERNEL=0`` for OFF), so the
only difference is the branching rule. Node counts are deterministic; wall is
reported but not used for the verdict (arms run in parallel workers).

Prints one JSON line per instance and the number of comparisons; exits
non-zero when zero comparisons ran or a certified objective disagrees.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from discopt_benchmarks.scripts import mpec_native_panel as mp
from discopt_benchmarks.scripts import relu_native_h2 as h2

ENVS = {
    "off": {"DISCOPT_COMPL_BRANCH": "0", "DISCOPT_NATIVE_SPATIAL_KERNEL": "0"},
    "on": {"DISCOPT_COMPL_BRANCH": "1"},
}


def run(src, argv, env):
    p = subprocess.run(
        [sys.executable, "-u", "-c", src, *argv],
        capture_output=True,
        text=True,
        env=dict(os.environ, **env),
        timeout=900,
    )
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError(p.stderr[-2000:])


def jobs():
    for n_in, w, d, cls, half in h2.PANEL:
        for seed in (0, 1):
            tag = f"relu-{cls}-{n_in}-{w}x{d}-s{seed}"
            argv = ["native0", str(n_in), str(w), str(d), cls, str(half), str(seed), "120"]
            yield tag, h2.ARM_SRC, argv
    for kind, seed, n in mp.PANEL:
        yield f"mpec-{kind}-s{seed}-n{n}", mp.ARM_SRC, ["native0", kind, str(seed), str(n), "60"]


def one(job):
    tag, src, argv = job
    return tag, {arm: run(src, argv, env) for arm, env in ENVS.items()}


def main() -> int:
    n_cmp = n_fail = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "3"))) as ex:
        for tag, res in ex.map(one, list(jobs())):
            on, off = res["on"], res["off"]
            fail = None
            if on["certified"] and off["certified"] and on["objective"] is not None:
                n_cmp += 1
                if abs(on["objective"] - off["objective"]) > 1e-4 * max(1.0, abs(off["objective"])):
                    fail = "certified objectives disagree"
                    n_fail += 1
            print(json.dumps({"instance": tag, "off": off, "on": on, "fail": fail}), flush=True)
    print(f"# comparisons made: {n_cmp}; failed: {n_fail}", flush=True)
    return 0 if n_cmp and not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
