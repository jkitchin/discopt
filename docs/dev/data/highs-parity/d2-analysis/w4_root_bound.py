"""D2 W4 entry probe: the Rust MILP driver's ROOT bound (after the root cut loop) per cut arm.

The bound of a ``max_nodes=2`` driver call is the root node's post-cut bound (``max_nodes=1``
returns before the root is evaluated: bound -inf). The driver is called directly through a
spy on ``discopt._rust.solve_milp_csc_py``, with the route's own arguments, once per arm.

Usage: python w4_root_bound.py mik-250-20-75-1,mik-250-20-75-5 [--arms gmi,cmir,gmi+cmir,none]
Prints one JSON line per (instance, arm) and an executed count; exits non-zero on 0 (§6).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
sys.path.insert(0, str(HERE))
os.environ.setdefault("DISCOPT_LP_MILP_BACKEND", "rust")

import lp_milp_loader as L  # noqa: E402
from d2_probe import instance_path  # noqa: E402

ARMS = {
    "none": dict(gmi_cuts=False, cmir_cuts=False),
    "gmi": dict(gmi_cuts=True, cmir_cuts=False),
    "cmir": dict(gmi_cuts=False, cmir_cuts=True),
    "gmi+cmir": dict(gmi_cuts=True, cmir_cuts=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instances")
    ap.add_argument("--arms", default="none,gmi,cmir,gmi+cmir")
    a = ap.parse_args()
    import discopt
    from discopt import _rust
    from discopt.modeling.core import from_nl

    assert discopt.__file__.startswith(str(REPO / "python")), discopt.__file__
    real = _rust.solve_milp_csc_py
    executed = 0
    for name in a.instances.split(","):
        captured: list = []

        def spy(*args, **kw):
            if not captured:
                captured.append((args, kw))
            return real(*args, **kw)

        _rust.solve_milp_csc_py = spy
        inst = L.read_instance(str(instance_path(name)))
        with tempfile.TemporaryDirectory() as td:
            nl = os.path.join(td, "m.nl")
            L.write_nl(inst, nl)
            m = from_nl(nl)
        m.solve(time_limit=2.0)
        _rust.solve_milp_csc_py = real
        args, kw = captured[0]
        for arm in a.arms.split(","):
            kw2 = dict(kw)
            kw2.update(ARMS[arm])
            a2 = list(args)
            a2[12] = 2  # max_nodes: the root node is evaluated
            out = real(*a2, **kw2)
            print(json.dumps({"name": name, "arm": arm, "root_bound": float(out[3])}), flush=True)
            executed += 1
    print(f"executed {executed}")
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
