#!/usr/bin/env python3
"""#2212: replay the captured nous1 undecided node LPs through the engine (one arm).

Each LP from `capture_nous1.py` is solved the way the spatial kernel solves it:
`solve_lp_warm_csc` from the basis it inherited, `warm_flip_boxed` on, default
options, no caller-side retry. Run once per arm, because the flag is a Rust
`OnceLock`::

    DISCOPT_LP_LOGICAL_DUAL_COLD=0 python -u replay_nous1.py IN.npz > off.jsonl
    DISCOPT_LP_LOGICAL_DUAL_COLD=1 python -u replay_nous1.py IN.npz > on.jsonl

HiGHS (the `throughput_panel.py` oracle settings) checks every verdict. #2212's bar
is that the engine decides >= 36 of 40 without a caller-side retry. One JSON line per
LP, then a summary and an executed count; exits non-zero on zero LPs or any
verdict HiGHS contradicts (CLAUDE.md §6).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from discopt._rust import profile_counters_py, profile_reset_py, solve_lp_warm_csc_py  # noqa: E402
from throughput_panel import highs, verdict  # noqa: E402  (also asserts the checkout)

arm = os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "0")
z = np.load(sys.argv[1])
n_lp = int(z["count"])
decided = wrong = 0
for i in range(n_lp):
    g = {k: z[f"{i}_{k}"] for k in ("m", "n", "col_ptr", "row_idx", "vals", "b", "l", "u", "c")}
    m, n = int(g["m"]), int(g["n"])
    f = {
        "A": sp.csc_matrix((g["vals"], g["row_idx"], g["col_ptr"]), shape=(m, n)),
        "m": m,
        "n": n,
        "c": g["c"].astype(float),
        "b": g["b"].astype(float),
        "l": g["l"].astype(float),
        "u": g["u"].astype(float),
    }
    ref = highs(f)
    profile_reset_py()
    t = time.perf_counter()
    status, _x, obj, iters, *_ = solve_lp_warm_csc_py(
        f["c"],
        m,
        n,
        g["col_ptr"].astype(np.int64),
        g["row_idx"].astype(np.int64),
        g["vals"].astype(float),
        f["b"],
        f["l"],
        f["u"],
        z[f"{i}_start_col_status"].astype(np.int8),
        z[f"{i}_start_basic_vars"].astype(np.int64),
        1e-9,
        100_000,
        None,
        True,
    )
    wall = time.perf_counter() - t
    ctr = profile_counters_py()
    r = {"status": status, "obj": float(obj), "iters": int(iters)}
    v = verdict(r, ref)
    decided += v == "decided-agree"
    wrong += v == "WRONG"
    row = {
        "lp": i,
        "m": m,
        "n": n,
        "arm": arm,
        "highs": ref,
        **r,
        "verdict": v,
        "wall": wall,
        "captured_status": str(z[f"{i}_status"]),
        "ctr": {k: v for k, v in ctr.items() if v and k.startswith("LogicalDualCold")},
    }
    print(json.dumps(row), flush=True)
print(f"# arm {arm}: decided {decided} / {n_lp}, WRONG {wrong}", flush=True)
print(f"EXECUTED: {n_lp}", flush=True)
sys.exit(1 if n_lp == 0 or wrong else 0)
